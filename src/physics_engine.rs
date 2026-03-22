use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SonarConfig {
    pub n_beams:     u32,
    pub n_rays:      u32,
    pub n_freq:      u32,
    pub sound_speed: f32,
    pub bandwidth:   f32,
    pub max_range:   f32,
    pub attenuation: f32,
    pub h_fov:       f32,
    pub v_fov:       f32,
    pub mu_default:  f32,
}

pub struct PhysicsInput<'a> {
    pub depth:          &'a [f32],
    pub normals:        &'a [f32],
    pub reflectivity:   &'a [f32],
    pub beam_corrector: &'a [f32],
    pub beam_corr_sum:  f32,
    pub frame:          u32,
    pub seed:           u32,
}

pub struct PhysicsOutput {
    pub intensity:  Vec<f32>,
    pub n_beams:    u32,
    pub n_freq:     u32,
    pub compute_ms: f64,
}

impl PhysicsOutput {
    pub fn peak_bin(&self) -> usize {
        let n  = self.n_freq  as usize;
        let nb = self.n_beams as usize;
        let mut bin_sum = vec![0.0f32; n];
        for b in 0..nb {
            for f in 0..n {
                bin_sum[f] += self.intensity[b * n + f];
            }
        }
        bin_sum.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BackscatterParams {
    n_beams:     u32,
    n_rays:      u32,
    n_freq:      u32,
    _pad0:       u32,
    sound_speed: f32,
    bandwidth:   f32,
    max_range:   f32,
    attenuation: f32,
    h_fov:       f32,
    v_fov:       f32,
    mu_default:  f32,
    _pad1:       f32,
    seed:        u32,
    frame:       u32,
    _pad2:       u32,
    _pad3:       u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulParams {
    n_beams:            u32,
    n_freq:             u32,
    beam_corrector_sum: f32,
    _pad:               u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParams {
    n_beams: u32,
    n_freq:  u32,
    log2_n:  u32,
    _pad:    u32,
}

pub struct SonarPhysicsEngine {
    device:           wgpu::Device,
    queue:            wgpu::Queue,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_bg:       wgpu::BindGroup,
    matmul_pipeline:  wgpu::ComputePipeline,
    matmul_bg_re:     wgpu::BindGroup,
    matmul_bg_im:     wgpu::BindGroup,
    fft_pipeline:     wgpu::ComputePipeline,
    fft_bg:           wgpu::BindGroup,
    depth_buf:        wgpu::Buffer,
    normal_buf:       wgpu::Buffer,
    refl_buf:         wgpu::Buffer,
    beam_corr_buf:    wgpu::Buffer,
    scatter_re_buf:   wgpu::Buffer,
    scatter_im_buf:   wgpu::Buffer,
    spectrum_re_buf:  wgpu::Buffer,
    spectrum_im_buf:  wgpu::Buffer,
    corrected_re_buf: wgpu::Buffer,
    corrected_im_buf: wgpu::Buffer,
    readback_buf:     wgpu::Buffer,
    scatter_uniform:  wgpu::Buffer,
    matmul_uniform:   wgpu::Buffer,
    config:           SonarConfig,
    adapter_name:     std::ffi::CString,
}

impl SonarPhysicsEngine {
    pub fn new(config: SonarConfig) -> Self {
        pollster::block_on(Self::init(config))
    }

    async fn init(config: SonarConfig) -> Self {
        assert!(
            config.n_freq.is_power_of_two() && config.n_freq <= 4096,
            "n_freq must be power-of-2 and <= 4096, got {}", config.n_freq
        );

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");

        let info = adapter.get_info();
        let adapter_name = std::ffi::CString::new(
            format!("{} ({:?})", info.name, info.backend)
        ).unwrap();
        println!("[sonar_physics] GPU: {}", adapter_name.to_str().unwrap());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_storage_size: 32768,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("device creation failed");

        let nb = config.n_beams as u64;
        let nr = config.n_rays  as u64;
        let nf = config.n_freq  as u64;

        let ray_f32    = nb * nr * 4;
        let ray_normal = nb * nr * 3 * 4;
        let beam_corr  = nb * nb * 4;
        let spectrum   = nb * nf * 4;

        let mk = |size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size, usage, mapped_at_creation: false,
            })
        };

        let S  = wgpu::BufferUsages::STORAGE;
        let CD = wgpu::BufferUsages::COPY_DST;
        let CS = wgpu::BufferUsages::COPY_SRC;
        let MR = wgpu::BufferUsages::MAP_READ;

        let depth_buf        = mk(ray_f32,    S | CD);
        let normal_buf       = mk(ray_normal, S | CD);
        let refl_buf         = mk(ray_f32,    S | CD);
        let beam_corr_buf    = mk(beam_corr,  S | CD);
        let scatter_re_buf   = mk(spectrum,   S | CS | CD);
        let scatter_im_buf   = mk(spectrum,   S | CS | CD);
        let spectrum_re_buf  = mk(spectrum,   S | CD);
        let spectrum_im_buf  = mk(spectrum,   S | CD);
        let corrected_re_buf = mk(spectrum,   S | CS | CD);
        let corrected_im_buf = mk(spectrum,   S | CS | CD);
        let readback_buf     = mk(spectrum,   CD | MR);

        let scatter_params = BackscatterParams {
            n_beams: config.n_beams, n_rays: config.n_rays,
            n_freq: config.n_freq, _pad0: 0,
            sound_speed: config.sound_speed, bandwidth: config.bandwidth,
            max_range: config.max_range, attenuation: config.attenuation,
            h_fov: config.h_fov, v_fov: config.v_fov,
            mu_default: config.mu_default, _pad1: 0.0,
            seed: 0, frame: 0, _pad2: 0, _pad3: 0,
        };
        let scatter_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&scatter_params),
            usage: wgpu::BufferUsages::UNIFORM | CD,
        });

        let log2_n = config.n_freq.trailing_zeros();
        let fft_params = FftParams {
            n_beams: config.n_beams, n_freq: config.n_freq, log2_n, _pad: 0
        };
        let fft_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&fft_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let matmul_params = MatmulParams {
            n_beams: config.n_beams, n_freq: config.n_freq,
            beam_corrector_sum: 1.0, _pad: 0,
        };
        let matmul_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&matmul_params),
            usage: wgpu::BufferUsages::UNIFORM | CD,
        });

        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backscatter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/backscatter.wgsl").into()),
        });
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul.wgsl").into()),
        });
        let fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fft.wgsl").into()),
        });

        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_u(0), bgl_r(1), bgl_r(2), bgl_r(3), bgl_rw(4), bgl_rw(5),
            ],
        });
        let scatter_pipeline = make_pipeline(&device, &scatter_bgl, &scatter_shader, "scatter");
        let scatter_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &scatter_bgl,
            entries: &[
                bge(0, scatter_uniform.as_entire_binding()),
                bge(1, depth_buf.as_entire_binding()),
                bge(2, normal_buf.as_entire_binding()),
                bge(3, refl_buf.as_entire_binding()),
                bge(4, scatter_re_buf.as_entire_binding()),
                bge(5, scatter_im_buf.as_entire_binding()),
            ],
        });

        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_u(0), bgl_r(1), bgl_r(2), bgl_rw(3)],
        });
        let matmul_pipeline = make_pipeline(&device, &matmul_bgl, &matmul_shader, "matmul");
        let matmul_bg_re = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &matmul_bgl,
            entries: &[
                bge(0, matmul_uniform.as_entire_binding()),
                bge(1, beam_corr_buf.as_entire_binding()),
                bge(2, spectrum_re_buf.as_entire_binding()),
                bge(3, corrected_re_buf.as_entire_binding()),
            ],
        });
        let matmul_bg_im = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &matmul_bgl,
            entries: &[
                bge(0, matmul_uniform.as_entire_binding()),
                bge(1, beam_corr_buf.as_entire_binding()),
                bge(2, spectrum_im_buf.as_entire_binding()),
                bge(3, corrected_im_buf.as_entire_binding()),
            ],
        });

        let fft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_u(0), bgl_rw(1), bgl_rw(2)],
        });
        let fft_pipeline = make_pipeline(&device, &fft_bgl, &fft_shader, "fft");
        let fft_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &fft_bgl,
            entries: &[
                bge(0, fft_uniform.as_entire_binding()),
                bge(1, corrected_re_buf.as_entire_binding()),
                bge(2, corrected_im_buf.as_entire_binding()),
            ],
        });

        let _ = fft_uniform; // suppress dead_code warning

        Self {
            device, queue,
            scatter_pipeline, scatter_bg,
            matmul_pipeline, matmul_bg_re, matmul_bg_im,
            fft_pipeline, fft_bg,
            depth_buf, normal_buf, refl_buf, beam_corr_buf,
            scatter_re_buf, scatter_im_buf,
            spectrum_re_buf, spectrum_im_buf,
            corrected_re_buf, corrected_im_buf,
            readback_buf,
            scatter_uniform, matmul_uniform,
            config, adapter_name,
        }
    }

    pub fn run(&self, input: &PhysicsInput<'_>) -> PhysicsOutput {
        pollster::block_on(self.dispatch(input)).expect("dispatch failed")
    }

    async fn dispatch(&self, input: &PhysicsInput<'_>) -> Result<PhysicsOutput, String> {
        let t0 = std::time::Instant::now();
        let nb = self.config.n_beams;
        let nr = self.config.n_rays;
        let nf = self.config.n_freq;
        let scale: f32 = 1048576.0;

        self.queue.write_buffer(&self.depth_buf,     0, bytemuck::cast_slice(input.depth));
        self.queue.write_buffer(&self.normal_buf,    0, bytemuck::cast_slice(input.normals));
        self.queue.write_buffer(&self.refl_buf,      0, bytemuck::cast_slice(input.reflectivity));
        self.queue.write_buffer(&self.beam_corr_buf, 0, bytemuck::cast_slice(input.beam_corrector));

        let sp = BackscatterParams {
            n_beams: nb, n_rays: nr, n_freq: nf, _pad0: 0,
            sound_speed: self.config.sound_speed,
            bandwidth:   self.config.bandwidth,
            max_range:   self.config.max_range,
            attenuation: self.config.attenuation,
            h_fov: self.config.h_fov, v_fov: self.config.v_fov,
            mu_default: self.config.mu_default, _pad1: 0.0,
            seed: input.seed, frame: input.frame, _pad2: 0, _pad3: 0,
        };
        self.queue.write_buffer(&self.scatter_uniform, 0, bytemuck::bytes_of(&sp));

        let mp = MatmulParams {
            n_beams: nb, n_freq: nf,
            beam_corrector_sum: input.beam_corr_sum, _pad: 0,
        };
        self.queue.write_buffer(&self.matmul_uniform, 0, bytemuck::bytes_of(&mp));

        // CMD1: clear + scatter + copy re to readback
        let mut e1 = self.device.create_command_encoder(&Default::default());
        e1.clear_buffer(&self.scatter_re_buf, 0, None);
        e1.clear_buffer(&self.scatter_im_buf, 0, None);
        {
            let mut p = e1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scatter"), timestamp_writes: None,
            });
            p.set_pipeline(&self.scatter_pipeline);
            p.set_bind_group(0, &self.scatter_bg, &[]);
            p.dispatch_workgroups((nb+7)/8, (nr+7)/8, 1);
        }
        e1.copy_buffer_to_buffer(&self.scatter_re_buf, 0, &self.readback_buf, 0, (nb*nf*4) as u64);
        self.queue.submit(std::iter::once(e1.finish()));

        let i32_re = self.rb_i32(&self.readback_buf, (nb*nf) as usize).await?;
        let f32_re: Vec<f32> = i32_re.iter().map(|&v| v as f32 / scale).collect();

        let mut e2 = self.device.create_command_encoder(&Default::default());
        e2.copy_buffer_to_buffer(&self.scatter_im_buf, 0, &self.readback_buf, 0, (nb*nf*4) as u64);
        self.queue.submit(std::iter::once(e2.finish()));
        let i32_im = self.rb_i32(&self.readback_buf, (nb*nf) as usize).await?;
        let f32_im: Vec<f32> = i32_im.iter().map(|&v| v as f32 / scale).collect();

        self.queue.write_buffer(&self.spectrum_re_buf, 0, bytemuck::cast_slice(&f32_re));
        self.queue.write_buffer(&self.spectrum_im_buf, 0, bytemuck::cast_slice(&f32_im));

        // CMD2: matmul + FFT + copy re to readback
        let mut e3 = self.device.create_command_encoder(&Default::default());
        {
            let mut p = e3.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_re"), timestamp_writes: None,
            });
            p.set_pipeline(&self.matmul_pipeline);
            p.set_bind_group(0, &self.matmul_bg_re, &[]);
            p.dispatch_workgroups((nf+15)/16, (nb+15)/16, 1);
        }
        {
            let mut p = e3.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_im"), timestamp_writes: None,
            });
            p.set_pipeline(&self.matmul_pipeline);
            p.set_bind_group(0, &self.matmul_bg_im, &[]);
            p.dispatch_workgroups((nf+15)/16, (nb+15)/16, 1);
        }
        {
            let mut p = e3.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft"), timestamp_writes: None,
            });
            p.set_pipeline(&self.fft_pipeline);
            p.set_bind_group(0, &self.fft_bg, &[]);
            p.dispatch_workgroups(nb, 1, 1);
        }
        e3.copy_buffer_to_buffer(&self.corrected_re_buf, 0, &self.readback_buf, 0, (nb*nf*4) as u64);
        self.queue.submit(std::iter::once(e3.finish()));
        let re_out = self.rb_f32(&self.readback_buf, (nb*nf) as usize).await?;

        let mut e4 = self.device.create_command_encoder(&Default::default());
        e4.copy_buffer_to_buffer(&self.corrected_im_buf, 0, &self.readback_buf, 0, (nb*nf*4) as u64);
        self.queue.submit(std::iter::once(e4.finish()));
        let im_out = self.rb_f32(&self.readback_buf, (nb*nf) as usize).await?;

        let intensity: Vec<f32> = re_out.iter().zip(im_out.iter())
            .map(|(re, im)| re*re + im*im)
            .collect();

        Ok(PhysicsOutput {
            intensity, n_beams: nb, n_freq: nf,
            compute_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn rb_i32(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<i32>, String> {
        let n = (count * 4) as u64;
        let slice = buf.slice(..n);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| e.to_string())?;
        let data = { let m = slice.get_mapped_range(); bytemuck::cast_slice::<u8,i32>(&m).to_vec() };
        buf.unmap();
        Ok(data)
    }

    async fn rb_f32(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        let n = (count * 4) as u64;
        let slice = buf.slice(..n);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| e.to_string())?;
        let data = { let m = slice.get_mapped_range(); bytemuck::cast_slice::<u8,f32>(&m).to_vec() };
        buf.unmap();
        Ok(data)
    }

    pub fn gpu_name(&self) -> &str {
        self.adapter_name.to_str().unwrap()
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label), layout: Some(&layout), module: shader,
        entry_point: "main", compilation_options: Default::default(),
    })
}

fn bgl_u(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false, min_binding_size: None,
        }, count: None,
    }
}
fn bgl_r(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false, min_binding_size: None,
        }, count: None,
    }
}
fn bgl_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false, min_binding_size: None,
        }, count: None,
    }
}
fn bge(binding: u32, resource: wgpu::BindingResource) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry { binding, resource }
}

#[no_mangle]
pub extern "C" fn sonar_physics_create(
    n_beams: u32, n_rays: u32, n_freq: u32,
    sound_speed: f32, bandwidth: f32,
    max_range: f32, attenuation: f32,
    h_fov: f32, v_fov: f32,
) -> *mut SonarPhysicsEngine {
    let config = SonarConfig {
        n_beams, n_rays, n_freq, sound_speed, bandwidth,
        max_range, attenuation, h_fov, v_fov, mu_default: 0.5,
    };
    Box::into_raw(Box::new(SonarPhysicsEngine::new(config)))
}

#[no_mangle]
pub unsafe extern "C" fn sonar_physics_destroy(engine: *mut SonarPhysicsEngine) {
    if !engine.is_null() { drop(Box::from_raw(engine)); }
}
