// sonar.wgsl — Simulates underwater acoustic backscatter noise
// Each ray distance gets perturbed by a deterministic "noise" value

struct SonarParams {
    noise_scale: f32,
    ray_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> ray_distances: array<f32>;
@group(0) @binding(1) var<uniform>             params: SonarParams;

// Pseudo-random noise based on index (no texture needed — fully deterministic)
fn sonar_noise(index: u32) -> f32 {
    let x = f32(index);
    // Simple hash-based noise mimicking acoustic backscatter variance
    return fract(sin(x * 127.1 + 311.7) * 43758.5453) * params.noise_scale;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.ray_count) {
        return;
    }
    // Add backscatter noise to each ray distance reading
    ray_distances[idx] = ray_distances[idx] + sonar_noise(idx);
}
