// =============================================================================
// fft.wgsl — Spectrum → Range Pass 3
//
// Applies an in-place Cooley-Tukey radix-2 FFT to each beam's spectrum.
//
// WHY FFT GIVES YOU RANGE:
//   After Pass 1, P_j(f) is the beam spectrum — the coherent sum of ray
//   contributions across frequency bins (Eq.8, Choi et al.).
//   The IFFT of P_j(f) gives p_j(t): the time-domain echo return signal.
//   Range maps to time as:   r = t · c / 2   (two-way travel)
//   So each output sample index t maps to range:
//       range_t = t · c / (2 · bandwidth)   [metres]
//   Output bin 0 = r=0 (sensor origin), bin N-1 = r≈max_range.
//
//   Example: c=1500 m/s, B=29.5 MHz → range resolution = c/(2B) ≈ 25 μm/bin
//   A flat floor at 10 m peaks at bin  t = 10 · 2B/c = 393333
//   (with n_freq=512 and appropriate bandwidth scaling)
//
// IMPLEMENTATION NOTES:
//   - Forward FFT with negative-exponent twiddle (matches scipy.fft.ifft sign
//     when input is conjugate-symmetric from the backscatter pass)
//   - One workgroup per beam: workgroup_size(256), dispatch(n_beams, 1, 1)
//   - Shared memory: 2 × 4096 × 4 bytes = 32 KB — fits in all major GPUs
//   - Power-of-2 constraint: n_freq must be ≤ 4096 and a power of 2
//     (limitation documented; CPU fallback for arbitrary sizes in host)
//   - Bit-reversal permutation precedes the butterfly stages
// =============================================================================

struct Params {
    n_beams:  u32,
    n_freq:   u32,
    log2_n:   u32,   // log2(n_freq) — precomputed by host
    _pad:     u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> p_re:   array<f32>; // [n_beams * n_freq]
@group(0) @binding(2) var<storage, read_write> p_im:   array<f32>;

// Workgroup scratchpad — 4096 × 4 bytes × 2 = 32 KB
// All threads in a workgroup share these for the butterfly passes.
var<workgroup> smem_re: array<f32, 4096>;
var<workgroup> smem_im: array<f32, 4096>;

const PI: f32 = 3.14159265358979323846;

// Bit-reversal: reverse the 'bits' lowest-order bits of v
fn bit_reverse(v: u32, bits: u32) -> u32 {
    var x = v;  var y = 0u;
    for (var i = 0u; i < bits; i++) {
        y = (y << 1u) | (x & 1u);
        x >>= 1u;
    }
    return y;
}

// Dispatch: workgroup_size(256, 1, 1), groups = (n_beams, 1, 1)
// Each workgroup processes one beam's entire spectrum.
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)         wg:  vec3<u32>,   // wg.x = beam index
    @builtin(local_invocation_id)  lid: vec3<u32>    // lid.x = thread 0..255
) {
    let beam = wg.x;
    if beam >= params.n_beams || params.n_freq > 4096u { return; }
    // NOTE: n_freq > 4096 returns silently — host must use CPU FFT for larger sizes.

    let n    = params.n_freq;
    let base = beam * n;

    // -------------------------------------------------------------------------
    // Step 1: Load + bit-reversal permutation into shared memory
    // Each thread loads multiple elements (stride 256) to cover all n up to 4096
    // -------------------------------------------------------------------------
    for (var i = lid.x; i < n; i += 256u) {
        let j = bit_reverse(i, params.log2_n);
        smem_re[j] = p_re[base + i];
        smem_im[j] = p_im[base + i];
    }
    workgroupBarrier();

    // -------------------------------------------------------------------------
    // Step 2: Cooley-Tukey butterfly iterations
    // log2(n) stages, each with n/2 butterfly operations.
    // Stage s combines pairs separated by stride = 2^(s+1).
    // Twiddle factor: W_N^k = e^{-i·2π·k/N} = cos(θ) - i·sin(θ)
    //   where θ = -2π·j / stride  (j = position within butterfly group)
    // -------------------------------------------------------------------------
    for (var s = 0u; s < params.log2_n; s++) {
        let half   = 1u << s;         // half-stride
        let stride = half << 1u;      // full stride = 2^(s+1)

        for (var i = lid.x; i < n / 2u; i += 256u) {
            let group = i / half;
            let j     = i % half;
            let i0    = group * stride + j;       // upper butterfly index
            let i1    = i0 + half;                // lower butterfly index

            // Twiddle factor W_N^j = e^{-i·2π·j/stride}
            let angle = -2.0 * PI * f32(j) / f32(stride);
            let wr = cos(angle);   let wi = sin(angle);

            // Complex multiply: t = W · smem[i1]
            let tr = smem_re[i1] * wr - smem_im[i1] * wi;
            let ti = smem_re[i1] * wi + smem_im[i1] * wr;

            // Butterfly combine: smem[i0] ± t
            let ur = smem_re[i0];  let ui = smem_im[i0];
            smem_re[i0] = ur + tr;  smem_im[i0] = ui + ti;
            smem_re[i1] = ur - tr;  smem_im[i1] = ui - ti;
        }
        workgroupBarrier();
    }

    // -------------------------------------------------------------------------
    // Step 3: Write back to global memory
    // Output p_re[beam,t] and p_im[beam,t] are the time-domain echo samples.
    // Intensity at sample t: I_t = p_re[t]² + p_im[t]²  (computed by host)
    // Range at sample t:     r_t = t · c / (2 · bandwidth)
    // -------------------------------------------------------------------------
    for (var i = lid.x; i < n; i += 256u) {
        p_re[base + i] = smem_re[i];
        p_im[base + i] = smem_im[i];
    }
}