struct SonarParams {
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    max_range: f32,
    step_size: f32,
    num_rays: u32,
    floor_height: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: SonarParams;
@group(0) @binding(1) var<storage, read_write> ray_distances: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let idx = id.x;
    if idx >= params.num_rays { return; }

    // spread rays across ~180 deg
    let angle = (f32(idx) / f32(params.num_rays)) * 3.14159265;

    let dir_x = cos(angle);
    let dir_y = -0.15;
    let dir_z = sin(angle);

    var x = params.origin_x;
    var y = params.origin_y;
    var z = params.origin_z;

    var hit = params.max_range;

    for (var step = 0; step < 512; step++) {

        x += dir_x * params.step_size;
        y += dir_y * params.step_size;
        z += dir_z * params.step_size;

        let dx = x - params.origin_x;
        let dy = y - params.origin_y;
        let dz = z - params.origin_z;

        let dist = sqrt(dx*dx + dy*dy + dz*dz);

        let floor =
            params.floor_height +
            sin(x * 0.3) * 2.0 +
            sin(z * 0.5) * 1.5;

        if y < floor {
            hit = dist;
            break;
        }

        if dist >= params.max_range {
            break;
        }
    }

    // small noise
    let noise =
        fract(sin(f32(idx) * 127.1 + 311.7) * 43758.5453) * 0.02;

    ray_distances[idx] = hit + noise;
}