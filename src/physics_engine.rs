// =============================================================================
// physics_engine.rs — wgpu Physics Pipeline
//
// Chains three GPU passes to implement Choi et al. (2021):
//   Pass 1 — backscatter.wgsl : Eq.14 scatter + Eq.8 phase sweep → P_j(f)
//   Pass 2 — matmul.wgsl      : beam pattern correction W·P / sum(W)
//   Pass 3 — fft.wgsl         : P_j(f) → p_j(t) → range image
//
// KEY DESIGN: all GPU buffers allocated once in SonarPhysicsEngine::new().
// Per-frame dispatch() only writes input data and submits three pipeline calls.
// This eliminates per-frame GPU allocation — the main bottleneck in the CUDA
// reference (identified in PR #29, GSoC 2025).
// =============================================================================

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// -----------------------------------------------------------------------------
// Public config — passed from Gazebo SDF or test harness
// -----------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SonarConfig {
    pub n_beams:     u32,   // number of sonar beams
    pub n_rays:      u32,   // rays per beam (from depth camera resolution)
    pub n_freq:      u32,   // FFT bins (must be power-of-2, ≤4096)
    pub sound_speed: f32,   // c [m/s]
    pub bandwidth:   f32,   // B [Hz]
    pub max_range:   f32,   // [m]
    pub attenuation: f32,   // α [Np/m]
    pub h_fov:       f32,   // horizontal FOV [radians]
    pub v_fov:       f32,   // vertical   FOV [radians]
    pub mu_default:  f32,   // default reflectivity (used if refl all-zero)
}

// -----------------------------------------------------------------------------
// Input per frame — matches Gazebo DepthCameraSensor output
// -----------------------------------------------------------------------------
pub struct PhysicsInput<'a> {
    pub depth:            &'a [f32],   // [n_beams * n_rays]  range per ray [m]
    pub normals:          &'a [f32],   // [n_beams * n_rays * 3]  surface normals
    pub reflectivity:     &'a [f32],   // [n_beams * n_rays]  μ per ray
    pub beam_corrector:   &'a [f32],   // [n_beams * n_beams] W matrix
    pub beam_corr_sum:    f32,         // sum of all W entries (for normalisation)
    pub frame:            u32,         // frame index → unique speckle per tick
    pub seed:             u32,         // run seed → repeatable across runs
}

// -----------------------------------------------------------------------------
// Output per frame
// -----------------------------------------------------------------------------
pub struct PhysicsOutput {
    pub intensity: Vec<f32>,   // [n_beams * n_freq]  |p_j(t)|²  range image
    pub n_beams:   u32,
    pub n_freq:    u32,
    pub compute_ms: f64,       // GPU time for this frame
}

impl PhysicsOutput {
    /// Range resolution: metres per FFT bin  r = t · c / (2·B)
    pub fn range_resolution(&self, config: &SonarConfig) -> f32 {
        config.sound_speed / (2.0 * config.bandwidth)
    }

    /// Bin index with highest total intensity across all beams — physical sanity check.
    /// For a flat floor at range R: expected_bin = R * 2 * B / c
    pub fn peak_bin(&self) -> usize {
        let n = self.n_freq as usize;
        let nb = self.n_beams as usize;
        let mut bin_sum = vec![0.0f32; n];
        for b in 0..nb {
            for f in 0..n {
                bin_sum[f] += self.intensity[b * n + f];
            }
        }
        bin_sum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// -----------------------------------------------------------------------------
// GPU-side uniform structs — must match shader layout exactly
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// The engine
// -----------------------------------------------------------------------------
pub struct SonarPhysicsEngine {
    device: wgpu::Device,
    queue:  wgpu::Queue,

    // Pass 1 — backscatter
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_bg:       wgpu::BindGroup,

    // Pass 2 — matmul
    matmul_pipeline:  wgpu::ComputePipeline,
    matmul_bg_re:     wgpu::BindGroup,   // runs twice: once for re, once for im
    matmul_bg_im:     wgpu::BindGroup,

    // Pass 3 — FFT
    fft_pipeline:     wgpu::ComputePipeline,
    fft_bg:           wgpu::BindGroup,

    // Persistent GPU buffers (allocated once, written each frame)
    depth_buf:        wgpu::Buffer,    // input
    normal_buf:       wgpu::Buffer,    // input
    refl_buf:         wgpu::Buffer,    // input
    beam_corr_buf:    wgpu::Buffer,    // input  (beam corrector matrix W)
    scatter_re_buf:   wgpu::Buffer,    // intermediate: atomic i32
    scatter_im_buf:   wgpu::Buffer,    // intermediate: atomic i32
    spectrum_re_buf:  wgpu::Buffer,    // intermediate: f32 after fixed-pt decode + matmul
    spectrum_im_buf:  wgpu::Buffer,
    corrected_re_buf: wgpu::Buffer,    // intermediate: after matmul
    corrected_im_buf: wgpu::Buffer,
    readback_buf:     wgpu::Buffer,    // CPU-readable copy of final intensity

    // Uniform buffers
    scatter_uniform:  wgpu::Buffer,
    matmul_uniform:   wgpu::Buffer,
    fft_uniform:      wgpu::Buffer,

    config: SonarConfig,
    adapter_name: std::ffi::CString,
}

impl SonarPhysicsEngine {
    pub fn new(config: SonarConfig) -> Self {
        pollster::block_on(Self::init(config))
    }

    async fn init(config: SonarConfig) -> Self {
        // Validate n_freq is power-of-2 and ≤ 4096
        assert!(
            config.n_freq.is_power_of_two() && config.n_freq <= 4096,
            "n_freq must be a power-of-2 and ≤ 4096, got {}",
            config.n_freq
        );

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("[sonar_physics] no GPU adapter found");

        let info = adapter.get_info();
        let adapter_name = std::ffi::CString::new(
            format!("{} ({:?})", info.name, info.backend)
        ).unwrap();
        println!("[sonar_physics] GPU: {}", adapter_name.to_str().unwrap());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        // 32KB workgroup memory for fft.wgsl smem_re + smem_im
                        max_compute_workgroup_storage_size: 32768,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("[sonar_physics] device creation failed");

        // ---- Buffer sizes ----
        let nb = config.n_beams as u64;
        let nr = config.n_rays  as u64;
        let nf = config.n_freq  as u64;

        let ray_f32   = nb * nr * 4;           // depth or refl
        let ray_normal = nb * nr * 3 * 4;      // normals (3 floats per ray)
        let beam_corr  = nb * nb * 4;          // W matrix
        let spectrum   = nb * nf * 4;          // f32 spectrum
        let spectrum_i32 = nb * nf * 4;        // atomic i32 spectrum

        // ---- Allocate all persistent GPU buffers once ----
        let mk_buf = |size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size, usage, mapped_at_creation: false,
            })
        };

        let depth_buf   = mk_buf(ray_f32,    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let normal_buf  = mk_buf(ray_normal, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let refl_buf    = mk_buf(ray_f32,    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let beam_corr_buf = mk_buf(beam_corr, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);

        // Atomic i32 output from scatter pass (cleared each frame via clear_buffer)
        let scatter_re_buf = mk_buf(spectrum_i32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let scatter_im_buf = mk_buf(spectrum_i32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);

        // f32 buffers for matmul input (decoded from fixed-point)
        let spectrum_re_buf  = mk_buf(spectrum, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let spectrum_im_buf  = mk_buf(spectrum, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);

        // Matmul output (also used as FFT in/out)
        let corrected_re_buf = mk_buf(spectrum, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let corrected_im_buf = mk_buf(spectrum, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);

        // CPU readback — final intensity [n_beams * n_freq]
        let readback_buf = mk_buf(spectrum, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);

        // ---- Uniform buffers ----
        let scatter_params = BackscatterParams {
            n_beams: config.n_beams, n_rays: config.n_rays, n_freq: config.n_freq, _pad0: 0,
            sound_speed: config.sound_speed, bandwidth: config.bandwidth,
            max_range: config.max_range, attenuation: config.attenuation,
            h_fov: config.h_fov, v_fov: config.v_fov,
            mu_default: config.mu_default, _pad1: 0.0,
            seed: 0, frame: 0, _pad2: 0, _pad3: 0,
        };
        let scatter_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&scatter_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let log2_n = config.n_freq.trailing_zeros();
        let fft_params = FftParams { n_beams: config.n_beams, n_freq: config.n_freq, log2_n, _pad: 0 };
        let fft_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&fft_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let matmul_params = MatmulParams {
            n_beams: config.n_beams, n_freq: config.n_freq,
            beam_corrector_sum: 1.0, // updated per frame
            _pad: 0,
        };
        let matmul_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&matmul_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Load shaders ----
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

        // ---- Build bind group layouts and pipelines ----

        // Pass 1 — backscatter
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_uniform(0),
                bgl_storage_r(1),
                bgl_storage_r(2),
                bgl_storage_r(3),
                bgl_storage_rw(4),
                bgl_storage_rw(5),
            ],
        });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&scatter_bgl], push_constant_ranges: &[],
            })),
            module: &scatter_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let scatter_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &scatter_bgl,
            entries: &[
                bg_entry(0, scatter_uniform.as_entire_binding()),
                bg_entry(1, depth_buf.as_entire_binding()),
                bg_entry(2, normal_buf.as_entire_binding()),
                bg_entry(3, refl_buf.as_entire_binding()),
                bg_entry(4, scatter_re_buf.as_entire_binding()),
                bg_entry(5, scatter_im_buf.as_entire_binding()),
            ],
        });

        // Pass 2 — matmul (two bind groups: re and im)
        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage_r(1), bgl_storage_r(2), bgl_storage_rw(3)],
        });
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&matmul_bgl], push_constant_ranges: &[],
            })),
            module: &matmul_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let matmul_bg_re = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &matmul_bgl,
            entries: &[
                bg_entry(0, matmul_uniform.as_entire_binding()),
                bg_entry(1, beam_corr_buf.as_entire_binding()),
                bg_entry(2, spectrum_re_buf.as_entire_binding()),
                bg_entry(3, corrected_re_buf.as_entire_binding()),
            ],
        });
        let matmul_bg_im = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &matmul_bgl,
            entries: &[
                bg_entry(0, matmul_uniform.as_entire_binding()),
                bg_entry(1, beam_corr_buf.as_entire_binding()),
                bg_entry(2, spectrum_im_buf.as_entire_binding()),
                bg_entry(3, corrected_im_buf.as_entire_binding()),
            ],
        });

        // Pass 3 — FFT
        let fft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_uniform(0), bgl_storage_rw(1), bgl_storage_rw(2)],
        });
        let fft_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&fft_bgl], push_constant_ranges: &[],
            })),
            module: &fft_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let fft_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &fft_bgl,
            entries: &[
                bg_entry(0, fft_uniform.as_entire_binding()),
                bg_entry(1, corrected_re_buf.as_entire_binding()),
                bg_entry(2, corrected_im_buf.as_entire_binding()),
            ],
        });

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
            scatter_uniform, matmul_uniform, fft_uniform,
            config, adapter_name,
        }
    }

    // -------------------------------------------------------------------------
    // run() — public API, wraps async dispatch
    // -------------------------------------------------------------------------
    pub fn run(&self, input: &PhysicsInput) -> PhysicsOutput {
        pollster::block_on(self.dispatch(input)).expect("dispatch failed")
    }

    async fn dispatch(&self, input: &PhysicsInput) -> Result<PhysicsOutput, String> {
        let t_start = std::time::Instant::now();

        let nb = self.config.n_beams;
        let nr = self.config.n_rays;
        let nf = self.config.n_freq;
        let scale: f32 = 1048576.0;  // must match SCALE in backscatter.wgsl

        // ---- 1. Upload input data to persistent GPU buffers ----
        self.queue.write_buffer(&self.depth_buf,     0, bytemuck::cast_slice(input.depth));
        self.queue.write_buffer(&self.normal_buf,    0, bytemuck::cast_slice(input.normals));
        self.queue.write_buffer(&self.refl_buf,      0, bytemuck::cast_slice(input.reflectivity));
        self.queue.write_buffer(&self.beam_corr_buf, 0, bytemuck::cast_slice(input.beam_corrector));

        // Update scatter uniform: frame + seed change each tick
        let scatter_params = BackscatterParams {
            n_beams: nb, n_rays: nr, n_freq: nf, _pad0: 0,
            sound_speed: self.config.sound_speed,
            bandwidth:   self.config.bandwidth,
            max_range:   self.config.max_range,
            attenuation: self.config.attenuation,
            h_fov: self.config.h_fov, v_fov: self.config.v_fov,
            mu_default: self.config.mu_default, _pad1: 0.0,
            seed: input.seed, frame: input.frame,
            _pad2: 0, _pad3: 0,
        };
        self.queue.write_buffer(&self.scatter_uniform, 0, bytemuck::bytes_of(&scatter_params));

        let matmul_params = MatmulParams {
            n_beams: nb, n_freq: nf,
            beam_corrector_sum: input.beam_corr_sum,
            _pad: 0,
        };
        self.queue.write_buffer(&self.matmul_uniform, 0, bytemuck::bytes_of(&matmul_params));

        // ---- 2. Build command buffer ----
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Clear atomic scatter buffers — MUST be zero before accumulation
        enc.clear_buffer(&self.scatter_re_buf, 0, None);
        enc.clear_buffer(&self.scatter_im_buf, 0, None);

        // --- Pass 1: backscatter (Eq.14 + Eq.8) ---
        // Each thread = one (beam, ray) pair
        // Workgroup (8,8): groups = (ceil(n_beams/8), ceil(n_rays/8), 1)
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass1_scatter"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.scatter_bg, &[]);
            pass.dispatch_workgroups((nb + 7) / 8, (nr + 7) / 8, 1);
        }

        // Decode fixed-point i32 → f32 on CPU (small buffer, fast)
        // We need to readback scatter_re/im, divide by SCALE, upload to spectrum buffers.
        // NOTE: in production Gazebo plugin this would be a 4th shader to avoid readback.
        // For the PoC we do it on CPU for simplicity and correctness transparency.
        enc.copy_buffer_to_buffer(
            &self.scatter_re_buf, 0,
            // Using corrected buffers as temporary staging (they'll be overwritten by matmul)
            &self.corrected_re_buf, 0,
            (nb * nf * 4) as u64,
        );
        enc.copy_buffer_to_buffer(
            &self.scatter_im_buf, 0,
            &self.corrected_im_buf, 0,
            (nb * nf * 4) as u64,
        );
        self.queue.submit(std::iter::once(enc.finish()));

        // Readback i32 scatter output, decode to f32, re-upload
        let i32_re = self.readback_i32(&self.corrected_re_buf, (nb * nf) as usize).await?;
        let i32_im = self.readback_i32(&self.corrected_im_buf, (nb * nf) as usize).await?;
        let f32_re: Vec<f32> = i32_re.iter().map(|&v| v as f32 / scale).collect();
        let f32_im: Vec<f32> = i32_im.iter().map(|&v| v as f32 / scale).collect();
        self.queue.write_buffer(&self.spectrum_re_buf, 0, bytemuck::cast_slice(&f32_re));
        self.queue.write_buffer(&self.spectrum_im_buf, 0, bytemuck::cast_slice(&f32_im));

        let mut enc2 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // --- Pass 2: beam correction matmul (re and im separately) ---
        // Workgroup (16,16): groups = (ceil(n_freq/16), ceil(n_beams/16), 1)
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass2_matmul_re"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &self.matmul_bg_re, &[]);
            pass.dispatch_workgroups((nf + 15) / 16, (nb + 15) / 16, 1);
        }
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass2_matmul_im"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &self.matmul_bg_im, &[]);
            pass.dispatch_workgroups((nf + 15) / 16, (nb + 15) / 16, 1);
        }

        // --- Pass 3: FFT  P_j(f) → p_j(t) ---
        // One workgroup per beam: dispatch(n_beams, 1, 1)
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass3_fft"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_pipeline);
            pass.set_bind_group(0, &self.fft_bg, &[]);
            pass.dispatch_workgroups(nb, 1, 1);
        }

        // Readback final re/im for intensity computation on CPU
        enc2.copy_buffer_to_buffer(
            &self.corrected_re_buf, 0, &self.readback_buf, 0,
            (nb * nf * 4) as u64,
        );
        self.queue.submit(std::iter::once(enc2.finish()));

        // ---- 3. Compute intensity |p(t)|² = re² + im² ----
        let re_out = self.readback_f32(&self.readback_buf, (nb * nf) as usize).await?;
        // For im we copy corrected_im_buf to readback (reuse readback_buf)
        let mut enc3 = self.device.create_command_encoder(&Default::default());
        enc3.copy_buffer_to_buffer(
            &self.corrected_im_buf, 0, &self.readback_buf, 0,
            (nb * nf * 4) as u64,
        );
        self.queue.submit(std::iter::once(enc3.finish()));
        let im_out = self.readback_f32(&self.readback_buf, (nb * nf) as usize).await?;

        let intensity: Vec<f32> = re_out.iter().zip(im_out.iter())
            .map(|(re, im)| re * re + im * im)
            .collect();

        let compute_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(PhysicsOutput { intensity, n_beams: nb, n_freq: nf, compute_ms })
    }

    // ---- Helpers ----

    async fn readback_i32(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<i32>, String> {
        let byte_len = (count * 4) as u64;
        let slice = buf.slice(..byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| e.to_string())?;
        let data = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, i32>(&mapped).to_vec()
        };
        buf.unmap();
        Ok(data)
    }

    async fn readback_f32(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        let byte_len = (count * 4) as u64;
        let slice = buf.slice(..byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| e.to_string())?;
        let data = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
        };
        buf.unmap();
        Ok(data)
    }

    pub fn gpu_name(&self) -> &str {
        self.adapter_name.to_str().unwrap()
    }
}

// ---- wgpu helper fns (reduce boilerplate) ----

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    }
}
fn bgl_storage_r(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    }
}
fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    }
}
fn bg_entry(binding: u32, resource: wgpu::BindingResource) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry { binding, resource }
}

// ---- C ABI — Gazebo plugin calls these ----

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