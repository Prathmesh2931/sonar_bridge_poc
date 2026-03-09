struct SonarParams {
    origin_x:     f32,
    origin_y:     f32,
    origin_z:     f32,
    max_range:    f32,
    step_size:    f32,
    num_rays:     u32,
    floor_height: f32,
    _pad:         f32,
}

@group(0) @binding(0) var<uniform>             params: SonarParams;
@group(0) @binding(1) var<storage, read_write> ray_distances: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_rays { return; }

    // Fan of rays spread across 180 degrees (sonar cone)
    let angle = (f32(idx) / f32(params.num_rays)) * 3.14159265;
    let dir_x  = cos(angle);
    let dir_y  = -0.15;        // slight downward tilt (sonar points down)
    let dir_z  = sin(angle);

    // Ray start position
    var pos_x = params.origin_x;
    var pos_y = params.origin_y;
    var pos_z = params.origin_z;

    var hit_dist = params.max_range; // default: no hit = max range

    // Ray marching loop — step forward until floor hit or max range
    for (var step = 0; step < 512; step++) {
        pos_x += dir_x * params.step_size;
        pos_y += dir_y * params.step_size;
        pos_z += dir_z * params.step_size;

        let travelled = sqrt(
            (pos_x - params.origin_x) * (pos_x - params.origin_x) +
            (pos_y - params.origin_y) * (pos_y - params.origin_y) +
            (pos_z - params.origin_z) * (pos_z - params.origin_z)
        );

        // Hit detection: ray went below floor
        if pos_y < params.floor_height {
            hit_dist = travelled;
            break;
        }

        // Exceeded max range
        if travelled >= params.max_range {
            break;
        }
    }

    // Gaussian-style acoustic noise (water turbidity + scattering)
    let noise = fract(sin(f32(idx) * 127.1 + 311.7) * 43758.5453) * 0.02;
    ray_distances[idx] = hit_dist + noise;
}
