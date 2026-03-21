// =============================================================================
// backscatter.wgsl — Choi et al. (2021) Physics Pass 1
//
// Implements:
//   Eq. 14  — scatter amplitude per ray
//             a_i = (ξ_xi + i·ξ_yi)/√2 · √(μ_i · cos²(α_i) · r_i² · dθ · dφ) · TL(r_i)
//
//   Eq. 8   — beam spectrum (coherent sum over all rays per beam per freq bin)
//             P_j(f) = S(f) · Σᵢ [ a_i · D(θᵢ,φᵢ) · e^(i·2·k·rᵢ) / rᵢ² ]
//
// Simplifications vs. full paper (acknowledged, not bugs):
//   - D(θ,φ) = 1  (ideal beam pattern, no sidelobes — full matrix applied in matmul.wgsl)
//   - S(f)   = 1  (flat source spectrum — window applied in host before FFT)
//   - TL folds one-way spreading + absorption; Eq.8 1/r² is absorbed into a_i
//
// Speckle model:
//   Each ray is an unresolved sub-wavelength scatterer. Its complex amplitude
//   (ξ_re + i·ξ_im) is drawn from CN(0,1) using Philox4x32-10 RNG, a direct
//   port of CUDA curand (Salmon et al. 2011, SC'11).
//   Summing many such scatterers gives Rayleigh-fading intensity — the
//   characteristic granular "speckle" of real sonar images.
//
// Output layout: out_re/out_im are atomic i32 buffers, [n_beams * n_freq].
//   Fixed-point scale = 1 << 20 = 1048576 to allow atomicAdd from f32.
//   Host divides by SCALE after readback to recover float values.
// =============================================================================

struct Params {
    n_beams:     u32,   // number of sonar beams  (NB)
    n_rays:      u32,   // rays per beam           (N)
    n_freq:      u32,   // FFT frequency bins      (Nf)
    _pad0:       u32,

    sound_speed: f32,   // c  [m/s]  — typically 1500
    bandwidth:   f32,   // B  [Hz]   — e.g. 29.5e6 for NPS sonar
    max_range:   f32,   // max valid ray distance  [m]
    attenuation: f32,   // α  [Np/m] absorption coeff (0 = lossless)

    h_fov:       f32,   // horizontal FOV [radians]  — total angular span
    v_fov:       f32,   // vertical   FOV [radians]
    mu_default:  f32,   // default reflectivity if refl buffer is uniform
    _pad1:       f32,

    seed:        u32,   // Philox RNG seed (changes per simulation run)
    frame:       u32,   // frame index     (changes per tick → unique speckle)
    _pad2:       u32,
    _pad3:       u32,
};

// Bindings
@group(0) @binding(0) var<uniform>              params:  Params;
@group(0) @binding(1) var<storage, read>        depth:   array<f32>; // [beam*n_rays + ray] metres
@group(0) @binding(2) var<storage, read>        normals: array<f32>; // [idx*3 + {x,y,z}] unit normals
@group(0) @binding(3) var<storage, read>        refl:    array<f32>; // [beam*n_rays + ray] μ ∈ [0,1]
@group(0) @binding(4) var<storage, read_write>  out_re:  array<atomic<i32>>; // [beam*n_freq + f]
@group(0) @binding(5) var<storage, read_write>  out_im:  array<atomic<i32>>; // [beam*n_freq + f]

const PI:    f32 = 3.14159265358979323846;
const SCALE: f32 = 1048576.0;   // 2^20 fixed-point scale for atomicAdd

// =============================================================================
// Philox4x32-10 RNG  (Salmon et al. 2011)
// Direct port of CUDA curand_philox4x32_10 — identical constants and rounds.
// Gives statistically independent streams per (frame, beam, ray) triple.
// =============================================================================
fn mulhilo32(a: u32, b: u32) -> vec2<u32> {
    // 32×32 → 64-bit multiply, split into hi and lo words
    let a_lo = a & 0xFFFFu;  let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;  let b_hi = b >> 16u;
    let p0   = a_lo * b_lo;  let p1   = a_hi * b_lo;
    let p2   = a_lo * b_hi;  let p3   = a_hi * b_hi;
    let mid  = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    let hi   = p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
    return vec2<u32>(hi, a * b);   // (hi, lo)
}

fn philox_round(c: vec4<u32>, k: vec2<u32>) -> vec4<u32> {
    let r0 = mulhilo32(0xD2511F53u, c.x);
    let r1 = mulhilo32(0xCD9E8D57u, c.z);
    return vec4<u32>(r1.x ^ c.y ^ k.x, r1.y, r0.x ^ c.w ^ k.y, r0.y);
}

fn philox4x32_10(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    // 10 Feistel rounds with linearly bumped key — same schedule as cuRAND
    var c = ctr;  var k = key;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);  k.x += 0x9E3779B9u;  k.y += 0xBB67AE85u;
    c = philox_round(c, k);
    return c;
}

fn u32_to_unit(x: u32) -> f32 {
    // Map u32 → (0, 1]  — avoids log(0) in Box-Muller below
    return f32(x) * (1.0 / 4294967296.0) + (0.5 / 4294967296.0);
}

fn philox_normal4(seed: u32, frame: u32, subseq: u32) -> vec4<f32> {
    // Generate 4 standard-normal floats via Box-Muller transform
    // Counter layout mirrors curand_init(seed, subseq, offset=0)
    let raw = philox4x32_10(
        vec4<u32>(frame, 0u, subseq, 0u),
        vec2<u32>(seed,  0u)
    );
    let u1 = u32_to_unit(raw.x);  let u2 = u32_to_unit(raw.y);
    let u3 = u32_to_unit(raw.z);  let u4 = u32_to_unit(raw.w);
    // Box-Muller: (u1,u2) → two independent N(0,1) values
    let r1 = sqrt(-2.0 * log(u1));  let r2 = sqrt(-2.0 * log(u3));
    return vec4<f32>(
        r1 * sin(2.0 * PI * u2),   // ξ_x1
        r1 * cos(2.0 * PI * u2),   // ξ_y1  ← used as ξ_xi, ξ_yi in Eq.14
        r2 * sin(2.0 * PI * u4),   // ξ_x2  (spare)
        r2 * cos(2.0 * PI * u4)    // ξ_y2  (spare)
    );
}

// =============================================================================
// Main compute kernel
// Dispatch: workgroup_size(8,8,1), groups = (ceil(n_beams/8), ceil(n_rays/8), 1)
// Each invocation handles one (beam, ray) pair.
// =============================================================================
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    let beam = gid.x;
    let ray  = gid.y;
    if beam >= params.n_beams || ray >= params.n_rays { return; }

    // -------------------------------------------------------------------------
    // 1. Read ray geometry from Gazebo depth buffer
    // -------------------------------------------------------------------------
    let idx = beam * params.n_rays + ray;
    let r   = depth[idx];                  // r_i: range to first intersection [m]

    // Skip rays that missed (r≤0) or exceeded max range
    if r <= 0.001 || r > params.max_range { return; }

    let nx = normals[idx * 3u + 0u];
    let ny = normals[idx * 3u + 1u];
    let nz = normals[idx * 3u + 2u];
    let mu = max(refl[idx], 0.0);          // μ_i: material reflectivity ∈ [0,1]

    // -------------------------------------------------------------------------
    // 2. Incidence angle cosine  cos(α_i) = |ray_dir · surface_normal|
    //    Ray direction in sensor frame from (beam_angle, ray_angle) pixel coords
    // -------------------------------------------------------------------------
    let beam_ang = params.h_fov * (f32(beam) / max(f32(params.n_beams) - 1.0, 1.0) - 0.5);
    let ray_ang  = params.v_fov * (f32(ray)  / max(f32(params.n_rays)  - 1.0, 1.0) - 0.5);
    // Unit ray direction (spherical coords → Cartesian, always length=1)
    let rd_x = sin(beam_ang) * cos(ray_ang);
    let rd_y = sin(ray_ang);
    let rd_z = cos(beam_ang) * cos(ray_ang);
    // Incidence angle: angle between incoming ray and surface normal
    // |dot(ray_dir, normal)| = cos(α_i)
    let cos_inc = abs(rd_x * nx + rd_y * ny + rd_z * nz);   // cos(α_i) in Eq.14

    // -------------------------------------------------------------------------
    // 3. Projected surface area per ray  (dA = r² · dθ_h · dθ_v)
    //    This is the solid-angle element each ray subtends at range r.
    // -------------------------------------------------------------------------
    let d_theta_h = params.h_fov / max(f32(params.n_beams) - 1.0, 1.0);
    let d_theta_v = params.v_fov / max(f32(params.n_rays)  - 1.0, 1.0);
    let dA = r * r * d_theta_h * d_theta_v;   // Eq.14: r_i² · dθ_i · dφ_i

    // -------------------------------------------------------------------------
    // 4. Transmission loss  TL(r) = e^{-α·r} / r
    //    Combines spherical spreading (1/r) and absorption (e^{-αr}).
    //    Two-way TL = TL² = e^{-2αr} / r² (folded into amplitude, see Eq.8).
    // -------------------------------------------------------------------------
    let TL = exp(-params.attenuation * r) / r;  // one-way; squared effect via a_i

    // -------------------------------------------------------------------------
    // 5. Eq. 14 — Scatter amplitude magnitude (deterministic part)
    //    a_i_det = √(μ_i) · |cos(α_i)| · √(dA) · TL
    //    Full Eq.14: a_i = (ξ_xi + i·ξ_yi)/√2 · √(μ_i·cos²(α_i)·r²·dθ·dφ) · TL
    //    → factored as: a_i = A_det · (ξ_xi + i·ξ_yi)/√2
    // -------------------------------------------------------------------------
    let A_det = sqrt(mu) * cos_inc * sqrt(dA) * TL;   // Eq.14 deterministic factor

    // -------------------------------------------------------------------------
    // 6. Coherent speckle: (ξ_xi + i·ξ_yi) ~ CN(0,1)
    //    Each ray is modelled as an unresolved scatterer with random phase.
    //    Summing many CN(0,1) variables → Rayleigh intensity distribution,
    //    which matches the granular appearance of real sonar images.
    //    Philox counter = (frame, 0, beam*n_rays+ray, 0) — unique per invocation.
    // -------------------------------------------------------------------------
    let subseq = beam * params.n_rays + ray;
    let xi     = philox_normal4(params.seed, params.frame, subseq);
    let xi_re  = xi.x / sqrt(2.0);    // ξ_xi / √2  (Eq.14 normalisation)
    let xi_im  = xi.y / sqrt(2.0);    // ξ_yi / √2

    // Complex scatter amplitude: a_i = A_det · (ξ_re + i·ξ_im)
    let amp_re = A_det * xi_re;
    let amp_im = A_det * xi_im;

    // -------------------------------------------------------------------------
    // 7. Eq. 8 — Phase sweep: accumulate into frequency bins
    //    P_j(f) = Σᵢ a_i · e^{i·2·k(f)·r_i}
    //    where k(f) = 2πf/c is the wavenumber at frequency f.
    //    e^{i·φ} = cos(φ) + i·sin(φ)   (Euler's formula)
    //    Contribution: (amp_re + i·amp_im) · (cos(φ) + i·sin(φ))
    //                = (amp_re·cos - amp_im·sin) + i·(amp_re·sin + amp_im·cos)
    //
    //    D(θ,φ) = 1 here (ideal beam). Full beam pattern correction is
    //    applied in matmul.wgsl (Pass 2) via the beam_corrector matrix.
    //    S(f) = 1 here (flat spectrum). Window shaping applied by host.
    // -------------------------------------------------------------------------
    let delta_f   = params.bandwidth / f32(params.n_freq);
    let n_freq_f  = f32(params.n_freq);
    let is_even   = (params.n_freq & 1u) == 0u;

    for (var f = 0u; f < params.n_freq; f++) {
        // Two-sided DC-centred frequency grid (matches numpy fftfreq convention)
        var freq: f32;
        if is_even {
            freq = delta_f * (-n_freq_f + 2.0 * (f32(f) + 1.0)) / 2.0;
        } else {
            freq = delta_f * (-(n_freq_f - 1.0) + 2.0 * (f32(f) + 1.0)) / 2.0;
        }

        // Wavenumber k = 2πf/c, two-way phase φ = 2·k·r  (Eq.8)
        let k   = 2.0 * PI * freq / params.sound_speed;
        let phi = 2.0 * r * k;   // two-way travel phase for this ray at freq f

        // Complex multiply: a_i · e^{iφ}
        let c = cos(phi);  let s = sin(phi);
        let contrib_re = amp_re * c - amp_im * s;
        let contrib_im = amp_re * s + amp_im * c;

        // Fixed-point atomic accumulation (all rays write to same bin in parallel)
        // Host recovers float by dividing readback by SCALE.
        let out_idx = beam * params.n_freq + f;
        let re_i32  = i32(clamp(contrib_re * SCALE, -2147483520.0, 2147483520.0));
        let im_i32  = i32(clamp(contrib_im * SCALE, -2147483520.0, 2147483520.0));
        atomicAdd(&out_re[out_idx], re_i32);
        atomicAdd(&out_im[out_idx], im_i32);
    }
}