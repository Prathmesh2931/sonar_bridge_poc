// lib.rs — Sonar compute engine with C-compatible FFI
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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

// ── Internal async GPU pipeline ───
async fn run_compute(data: &mut [f32]) -> Result<(), String> {

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::HighPerformance,
            compatible_surface:     None,
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

    let ray_count = data.len() as u32;
    let byte_len  = (data.len() * std::mem::size_of::<f32>()) as u64;

    // Upload ray buffer to GPU
    let storage_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("ray_distances"),
        contents: bytemuck::cast_slice(data),
        usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Staging buffer for GPU → CPU readback
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("readback"),
        size:               byte_len,
        usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Upload sonar physics params
    let params = SonarParams {
        origin_x:     0.0,   // sonar at world origin
        origin_y:     5.0,   // 5 metres above floor
        origin_z:     0.0,
        max_range:    50.0,  // 50 metre sonar range
        step_size:    0.05,  // 5 cm per ray step
        num_rays:     ray_count,
        floor_height: 0.0,   // floor at y=0
        _pad:         0.0,
    };

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
    });

    // Compile shader
    let shader_src = include_str!("../shaders/sonar.wgsl");
    let shader     = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("sonar_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // Bind group layout
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("sonar_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
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
            label:               Some("sonar_pipeline"),
            layout:              Some(&pipeline_layout),
            module:              &shader,
            entry_point:         "main",
            compilation_options: Default::default(),
        });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("sonar_bg"),
        layout:  &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding:  1,
                resource: storage_buf.as_entire_binding(),
            },
        ],
    });

    // Encode + dispatch
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("sonar_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = (ray_count + 63) / 64;
        pass.dispatch_workgroups(groups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&storage_buf, 0, &readback_buf, 0, byte_len);
    queue.submit(std::iter::once(encoder.finish()));

    // Read back results to CPU
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
pub fn process_rays(data: &mut [f32]) {
    pollster::block_on(run_compute(data)).expect("wgpu compute failed");
}

// ── C FFI surface (called from C++ / Gazebo plugin) ──────────────────────────
#[no_mangle]
pub unsafe extern "C" fn process_sonar_data(data: *mut f32, len: usize) {
    assert!(!data.is_null(), "process_sonar_data: null pointer");
    let slice = unsafe { std::slice::from_raw_parts_mut(data, len) };
    process_rays(slice);
}

#[no_mangle]
pub extern "C" fn sonar_backend_name() -> *const std::ffi::c_char {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let name = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference:       wgpu::PowerPreference::HighPerformance,
        compatible_surface:     None,
        force_fallback_adapter: false,
    }))
    .map(|a| a.get_info().name)
    .unwrap_or_else(|| "unknown".to_string());

    let cstr = std::ffi::CString::new(name).unwrap();
    cstr.into_raw()
}
