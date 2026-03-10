use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const MAX_RAYS: usize = 262144;
const F32_SIZE: usize = std::mem::size_of::<f32>();

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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

pub struct SonarEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    storage_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    adapter_name: std::ffi::CString,
}

impl SonarEngine {

    pub fn new() -> Self {
        pollster::block_on(Self::init())
    }

    async fn init() -> Self {

        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no gpu adapter");

        let info = adapter.get_info();

        let adapter_name = std::ffi::CString::new(
            format!("{} ({:?})", info.name, info.backend)
        ).unwrap();

        println!("sonar gpu: {}", adapter_name.to_str().unwrap());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sonar.wgsl").into()
            ),
        });

        let byte_len = (MAX_RAYS * F32_SIZE) as u64;

        let storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let params = SonarParams {
            origin_x: 0.0,
            origin_y: 5.0,
            origin_z: 0.0,
            max_range: 50.0,
            step_size: 0.05,
            num_rays: 1024,
            floor_height: 0.0,
            _pad: 0.0,
        };

        let uniform_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            }
        );

        let bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }
        );

        let layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
            }
        );

        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: storage_buf.as_entire_binding(),
                    },
                ],
            }
        );

        Self {
            device,
            queue,
            pipeline,
            bind_group,
            storage_buf,
            readback_buf,
            uniform_buf,
            adapter_name,
        }
    }

    pub fn process(&self, data: &mut [f32]) {

        assert!(data.len() <= MAX_RAYS);

        pollster::block_on(self.dispatch(data))
            .expect("dispatch failed");
    }

    async fn dispatch(&self, data: &mut [f32]) -> Result<(), String> {

        let ray_count = data.len() as u32;
        let byte_len = (data.len() * F32_SIZE) as u64;

        let params = SonarParams {
            origin_x: 0.0,
            origin_y: 5.0,
            origin_z: 0.0,
            max_range: 50.0,
            step_size: 0.05,
            num_rays: ray_count,
            floor_height: 0.0,
            _pad: 0.0,
        };

        self.queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&params),
        );

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None }
        );

        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                }
            );

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((ray_count + 63) / 64, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.storage_buf,
            0,
            &self.readback_buf,
            0,
            byte_len,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.readback_buf.slice(..byte_len);

        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv().unwrap().map_err(|e| e.to_string())?;

        {
            let mapped = slice.get_mapped_range();
            data.copy_from_slice(bytemuck::cast_slice(&mapped));
        }

        self.readback_buf.unmap();

        Ok(())
    }

    pub fn gpu_name(&self) -> &str {
        self.adapter_name.to_str().unwrap()
    }
}

#[no_mangle]
pub extern "C" fn sonar_engine_init() -> *mut SonarEngine {
    Box::into_raw(Box::new(SonarEngine::new()))
}

#[no_mangle]
pub unsafe extern "C" fn sonar_engine_update(
    engine: *mut SonarEngine,
    data: *mut f32,
    len: usize,
) {

    if engine.is_null() || data.is_null() {
        return;
    }

    let slice = std::slice::from_raw_parts_mut(data, len);
    (*engine).process(slice);
}

#[no_mangle]
pub unsafe extern "C" fn sonar_engine_destroy(engine: *mut SonarEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

#[no_mangle]
pub unsafe extern "C" fn sonar_backend_name(
    engine: *mut SonarEngine,
) -> *const std::ffi::c_char {

    if engine.is_null() {
        return std::ptr::null();
    }

    (*engine).adapter_name.as_ptr()
}