// =============================================================================
// matmul.wgsl — Beam Correction Pass 2
//
// Implements the beam pattern correction from Choi et al. (2021).
// The beam corrector matrix W encodes the directional sensitivity D(θ,φ)
// of each beam — it redistributes energy between adjacent beams to account
// for sidelobe leakage (the D(θ,φ) term that is set to 1 in backscatter.wgsl).
//
// Operation:
//   P_corrected[beam, f] = Σₖ W[beam, k] · P_raw[k, f] / beam_corrector_sum
//
// This is a standard GEMM: C = A·B / norm
//   A = beam_corrector [n_beams × n_beams]
//   B = spectrum_re/im  [n_beams × n_freq]   (output from backscatter pass)
//   C = corrected       [n_beams × n_freq]
//
// Tiled 16×16 shared-memory GEMM replaces cuBLAS SGEMM from CUDA reference.
// The beam_corrector_sum normalisation preserves total acoustic energy.
// Run separately for real and imaginary parts (two dispatches from host).
// =============================================================================

struct Params {
    n_beams:            u32,
    n_freq:             u32,
    beam_corrector_sum: f32,   // normalisation: sum of all W entries
    _pad:               u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       A:      array<f32>;  // beam_corrector [n_beams*n_beams]
@group(0) @binding(2) var<storage, read>       B:      array<f32>;  // spectrum input  [n_beams*n_freq]
@group(0) @binding(3) var<storage, read_write> C:      array<f32>;  // output          [n_beams*n_freq]

// Shared memory tiles — 16×16 f32 each = 1KB per tile, 2KB total
// Well within the 32KB workgroup memory limit on all major GPU vendors.
var<workgroup> tile_A: array<array<f32, 16>, 16>;
var<workgroup> tile_B: array<array<f32, 16>, 16>;

// Dispatch: workgroup_size(16,16,1), groups = (ceil(n_freq/16), ceil(n_beams/16), 1)
// Thread (lid.x, lid.y) computes C[row=gid.y, col=gid.x]
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>
) {
    let row = gid.y;   // output beam index
    let col = gid.x;   // output freq  index
    let in_bounds = (row < params.n_beams) && (col < params.n_freq);

    var acc: f32 = 0.0;
    let num_tiles = (params.n_beams + 15u) / 16u;

    // Classic tiled GEMM: iterate over 16-wide tiles along the k dimension
    for (var t = 0u; t < num_tiles; t++) {
        // Load A[row, t*16 + lid.x] into shared tile
        let k_a = t * 16u + lid.x;
        if in_bounds && k_a < params.n_beams {
            tile_A[lid.y][lid.x] = A[row * params.n_beams + k_a];
        } else {
            tile_A[lid.y][lid.x] = 0.0;
        }
        // Load B[t*16 + lid.y, col] into shared tile
        let k_b = t * 16u + lid.y;
        if in_bounds && k_b < params.n_beams {
            tile_B[lid.y][lid.x] = B[k_b * params.n_freq + col];
        } else {
            tile_B[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();

        // Accumulate partial dot product for this tile
        for (var k = 0u; k < 16u; k++) {
            acc += tile_A[lid.y][k] * tile_B[k][lid.x];
        }
        workgroupBarrier();
    }

    // Write normalised result: divide by beam_corrector_sum to preserve energy
    let norm = max(params.beam_corrector_sum, 1e-12);
    if in_bounds {
        C[row * params.n_freq + col] = acc / norm;
    }
}