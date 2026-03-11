# sonar_bridge_poc

Prototype for a **GPU-accelerated sonar compute pipeline** using Rust (`wgpu`)
with a C++ interface compatible with Gazebo sensor plugins.

Built to validate the architecture needed for a vendor-agnostic sonar
implementation before writing a GSoC proposal.

---

## Key Highlights

- **Vendor agnostic:** zero CUDA dependency — same binary targets Vulkan (Linux), Metal (macOS), DX12 (Windows)
- **Simulator ready:** FFI lifecycle mirrors official `gz::sensors::GpuLidarSensor` (`Load` → `Update` → `Unload`)
- **Physics grounded:** development guided by NPS multibeam sonar `sonar_calculation_cuda.cu` reference
- **Zero allocation per frame:** GPU buffers pre-allocated at `Load()`, only `queue.write_buffer()` called per `Update()`
- **0.09ms steady-state dispatch** at 1024 rays — under 1% of Gazebo's 16.6ms 60Hz frame budget

---

## What This Validates

`sonar_bridge_poc` tests the core architectural bet of the GSoC project: that a
Rust/wgpu compute pipeline can sit behind a stable C ABI and serve as a drop-in
GPU backend for a C++ Gazebo sensor plugin.

The engine exposes three `extern "C"` symbols:
```c
sonar_engine_init()    // → gazebo::SensorPlugin::Load()
sonar_engine_update()  // → gazebo::SensorPlugin::Update()
sonar_engine_destroy() // → gazebo::SensorPlugin::Unload()
```

The GPU context (`wgpu::Device`, `wgpu::Queue`, compute pipeline) is created
once in `init()` and held persistently in the `SonarEngine` struct.
Per-frame `Update()` calls only invoke `queue.write_buffer()` on a pre-allocated
uniform buffer and dispatch the compute shader — **zero GPU memory allocation
per frame**.

On an RTX 3050 at 1024 rays, steady-state dispatch cost is **0.09ms**, well
within the 16.6ms Gazebo 60Hz budget.

---

## Architectural Alignment with Gazebo Sensors

The `sonar_bridge_poc` lifecycle is directly modelled on
`gz::sensors::GpuLidarSensor` from `gazebosim/gz-sensors`:

| Gazebo GpuLidarSensor | sonar_bridge_poc |
|---|---|
| `Load()` — allocates `laserBuffer`, sets up rendering | `sonar_engine_init()` — allocates GPU buffers, compiles shader |
| `Update()` → `Render()` → `ApplyNoise()` → `Publish()` | `sonar_engine_update()` — dispatch → readback → caller publishes |
| `OnNewLidarFrame()` receives `const float* _scan` | FFI accepts `float* data, size_t len` — same pattern |
| Buffer checked for null, reused if exists | `MAX_RAYS` buffer allocated once, reused every frame |

The key difference: `GpuLidarSensor` copies rendering output directly.
The SONAR plugin needs an extra compute pass between render and publish —
acoustic physics processing — which is what the wgpu pipeline provides.

---

## Architecture
```
C++ Host / Gazebo Plugin
        │
        │  sonar_engine_init()    ← Load()
        │  sonar_engine_update()  ← Update()
        │  sonar_engine_destroy() ← Unload()
        ▼
Rust FFI Layer  (src/lib.rs — SonarEngine struct)
        ▼
wgpu Compute Pipeline
        ▼
WGSL Shader  (shaders/sonar.wgsl)
ray marching + sine-wave bathymetry + acoustic noise
        ▼
GPU — Vulkan (Linux) / Metal (macOS) / DX12 (Windows)
Intel / NVIDIA / AMD — zero code changes between vendors
```

---

## Hardware Compatibility

| Platform | Backend | Status |
|----------|---------|--------|
| Linux    | Vulkan  | ✅ verified — RTX 3050 + Intel RPL-P |
| macOS    | Metal   | ✅ wgpu-supported, untested on this machine |
| Windows  | DX12    | ✅ wgpu-supported, untested on this machine |

Enabled by `wgpu::Backends::all()` — no `#ifdef`, no vendor SDK.

---

## Project Structure
```
sonar_bridge_poc/
├── src/
│   ├── lib.rs              — SonarEngine struct + C FFI exports
│   └── main.rs             — benchmark runner
├── shaders/
│   └── sonar.wgsl          — WGSL compute shader
├── cpp_host/
│   ├── sonar_engine.h      — C header for Gazebo plugin integration
│   └── main.cpp            — Gazebo lifecycle simulation
├── gazebo_plugin/
│   └── SonarPlugin.cpp     — plugin stub (no Gazebo install needed)
├── visualize.py            — Python heatmap of sonar output
└── CMakeLists.txt          — ament_cmake compatible build
```

---

## Build and Run

**One-time setup (Ubuntu):**
```bash
sudo apt install -y build-essential libvulkan1 vulkan-tools
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**Run:**
```bash
bash build_and_run.sh
```

---

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU, Vulkan backend
```
GPU init — one-time cost at Gazebo Load():  2.16s

Dispatch cost per Update() call:
┌──────────────┬──────────────────────┐
│  Ray Count   │  Dispatch Time (ms)  │
├──────────────┼──────────────────────┤
│          256 │                 0.29 │
│         1024 │                 0.11 │
│         4096 │                 0.10 │
│        16384 │                 0.14 │
│        65536 │                 0.86 │
│       262144 │                 1.33 │
└──────────────┴──────────────────────┘

Steady state at 1024 rays:  0.085ms
Gazebo 60Hz frame budget:   16.6ms
Sonar compute usage:        < 1% of frame budget
```

![Sonar heatmap and benchmark](sonar_demo.png)

---

## Gazebo Lifecycle Test
```bash
./gazebo_plugin_test
```
```
[SonarPlugin::Load]   GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Vulkan)
[SonarPlugin::Load]   Ready. Rays: 1024

[SonarPlugin::Update] 0.27ms | rays[0]=23.97m  rays[511]=50.00m  rays[1023]=50.01m
[SonarPlugin::Update] 0.10ms | rays[0]=23.97m  rays[511]=50.00m  rays[1023]=50.01m
[SonarPlugin::Update] 0.09ms | rays[0]=23.97m  rays[511]=50.00m  rays[1023]=50.01m
[SonarPlugin::Update] 0.09ms | rays[0]=23.97m  rays[511]=50.00m  rays[1023]=50.01m
[SonarPlugin::Update] 0.09ms | rays[0]=23.97m  rays[511]=50.00m  rays[1023]=50.01m

[SonarPlugin::Unload] GPU context released.
```

Tick 1 slower = GPU pipeline JIT compilation. Ticks 2-5 = production steady state.

---

## Current Shader Physics

Each GPU thread = one sonar ray (`global_invocation_id`).

**180-degree sonar fan:**
```wgsl
let angle = (f32(idx) / f32(params.num_rays)) * 3.14159265;
```

**Sine-wave bathymetry** (produces varying per-ray distances):
```wgsl
let floor_at = params.floor_height
             + sin(pos_x * 0.3) * 2.0
             + sin(pos_z * 0.5) * 1.5;
```

**Acoustic scattering noise** (deterministic hash, 2cm variance at 50m):
```wgsl
let noise = fract(sin(f32(idx) * 127.1 + 311.7) * 43758.5453) * 0.02;
```

**Physics gap vs full NPS model** (`sonar_calculation_cuda.cu`):

| Missing component | Formula |
|---|---|
| Lambert backscatter | `I ∝ sqrt(μ · cos(θ))` |
| Propagation loss | `TL = (1/r²) · e^{-2αr}` |
| Frequency-domain echo summation | spectral bin accumulation |
| Beam-pattern correction | sinc `beamCorrector` matrix multiply |
| Range cell conversion | inverse FFT (`cufftExecC2C` → Cooley-Tukey) |

---

## GSoC Deliverable — 4-Pass WGSL Pipeline

Full GSoC work ports the NPS CUDA pipeline to vendor-agnostic WGSL:

**Pass 0 — 3D Volumetric Beam Projection**
Upgrade current 2D horizontal fan to a full 3D cone.
Each thread computes ray direction from azimuth + elevation indices:
```
dir = (cos(elev)·cos(az), sin(elev), cos(elev)·sin(az))
dispatch: (beam_count/64, elev_count, 1)
```
Horizontal FOV and vertical FOV configurable via uniform params.

**Pass 1 — Lambert scatter + propagation loss**
Port of `sonar_calculation_cuda.cu` scatter kernel.
- Backscatter: `sqrt(μ · cos(θ))` where θ from Gazebo normal map
- Propagation: `(1/r²) · e^{-2αr}` with absorption coeff α (salinity + frequency)
- Input: depth image + normal image from Gazebo `DepthCameraSensor`

**Pass 2 — Ray summation**
Parallel column reduction porting `column_sums_reduce`.
Sums ray contributions per beam across aperture.

**Pass 3 — Beam correction**
Matrix multiply with pre-computed sinc `beamCorrector`.
Port of `gpu_matrix_mult`. Loaded once at `Load()`, reused every frame.

**Pass 4 — Batched FFT**
Cooley-Tukey FFT replacing `cufftExecC2C`.
Converts spectral bins to final sonar range-cell image.

All input buffers (depth, normals, noise, reflectivity) from
Gazebo `DepthCameraSensor` into persistent `wgpu::Buffer`s —
same single-allocation pattern as NPS host plugin
(`rand_image`, `window`, `beamCorrector` in `Load()`).

---

## Project References

- **Lifecycle validation:** [gazebosim/gz-sensors](https://github.com/gazebosim/gz-sensors) — `GpuLidarSensor.cc`
- **Acoustic physics:** [nps_uw_multibeam_sonar](https://github.com/Field-Robotics-Lab/nps_uw_multibeam_sonar) — `sonar_calculation_cuda.cu`
- **Compute pipeline:** [Learn wgpu](https://sotrh.github.io/learn-wgpu/) — compute shader resource management

---

## License
MIT
