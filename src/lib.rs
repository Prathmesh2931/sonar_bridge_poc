// lib.rs — Sonar compute engine with C-compatible FFI
//
// Architecture:
//   C++ (Gazebo plugin)
//     └─ calls process_sonar_data()  [extern "C"]
//          └─ Rust initialises wgpu (Vulkan / Metal / DX12 / WebGPU)
//               └─ uploads ray buffer → GPU compute shader
//                    └─ downloads result → writes back into the same C pointer

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ── Uniform struct mirroring the WGSL SonarParams ────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SonarParams {
    noise_scale: f32,
    ray_count:   u32,
    _pad0:       u32,
    _pad1:       u32,
}

// ── Internal async implementation ─────────────────────────────────────────────
async fn run_compute(data: &mut [f32], noise_scale: f32) -> Result<(), String> {
    // 1. Initialise wgpu — picks the best available backend automatically
    //    (Vulkan on Linux/Windows, Metal on macOS, DX12 fallback on Windows)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(), // ← vendor-agnostic: Intel, NVIDIA, AMD, etc.
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::HighPerformance,
            compatible_surface:     None, // headless — no window needed
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No GPU adapter found")?;

    let adapter_info = adapter.get_info();
    println!(
        "[sonar_engine] GPU: {} | Backend: {:?}",
        adapter_info.name, adapter_info.backend
    );

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(|e| e.to_string())?;

    // 2. Upload ray-distance data to GPU
    let ray_count   = data.len() as u32;
    let byte_len    = (data.len() * std::mem::size_of::<f32>()) as u64;

    let storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("ray_distances"),
        contents: bytemuck::cast_slice(data),
        usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Read-back (staging) buffer — GPU→CPU copy target
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("readback"),
        size:               byte_len,
        usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 3. Upload uniform params
    let params = SonarParams {
        noise_scale,
        ray_count,
        _pad0: 0,
        _pad1: 0,
    };

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
    });

    // 4. Compile WGSL shader (embedded at compile time — no runtime file I/O)
    let shader_src = include_str!("../shaders/sonar.wgsl");
    let shader     = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("sonar_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // 5. Build bind-group layout + pipeline
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("sonar_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count:      None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count:      None,
                },
            ],
        });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("sonar_pl"),
        bind_group_layouts:   &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("sonar_pipeline"),
            layout:      Some(&pipeline_layout),
            module:      &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("sonar_bg"),
        layout:  &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding:  0,
                resource: storage_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding:  1,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    });

    // 6. Encode & dispatch
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:              Some("sonar_pass"),
            timestamp_writes:   None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // workgroup_size = 64 (matches WGSL), ceil(ray_count / 64) groups
        let groups = (ray_count + 63) / 64;
        pass.dispatch_workgroups(groups, 1, 1);
    }

    // 7. Copy GPU storage → staging buffer
    encoder.copy_buffer_to_buffer(&storage_buf, 0, &readback_buf, 0, byte_len);
    queue.submit(std::iter::once(encoder.finish()));

    // 8. Map staging buffer back to CPU and write into caller's slice
    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().map_err(|e| e.to_string())?;

    let mapped = slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&mapped);
    data.copy_from_slice(result);

    Ok(())
}

// ── Public Rust API ───────────────────────────────────────────────────────────
pub fn process_rays(data: &mut [f32], noise_scale: f32) {
    pollster::block_on(run_compute(data, noise_scale))
        .expect("wgpu compute failed");
}

// ── C FFI surface (called from C++ / Gazebo plugin) ──────────────────────────
/// # Safety
/// `data` must point to a valid array of `len` f32 values.
#[no_mangle]
pub unsafe extern "C" fn process_sonar_data(data: *mut f32, len: usize) {
    assert!(!data.is_null(), "process_sonar_data: null pointer");
    let slice = unsafe { std::slice::from_raw_parts_mut(data, len) };
    process_rays(slice, 0.05); // 5 cm acoustic noise — tunable
}

/// Returns the wgpu backend name as a null-terminated C string.
/// The caller must NOT free this pointer (it points to a static string).
#[no_mangle]
pub extern "C" fn sonar_backend_name() -> *const std::ffi::c_char {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    // Best-effort sync adapter probe
    let name = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference:       wgpu::PowerPreference::HighPerformance,
        compatible_surface:     None,
        force_fallback_adapter: false,
    }))
    .map(|a| a.get_info().name)
    .unwrap_or_else(|| "unknown".to_string());

    // Leak is intentional — static lifetime for the C caller
    let cstr = std::ffi::CString::new(name).unwrap();
    cstr.into_raw()
}
