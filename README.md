cat > ~/gsoc2026/sonar_bridge_poc/README.md << 'EOF'
# sonar_bridge_poc

Prototype for a **GPU-accelerated sonar compute pipeline** using Rust (wgpu)
with a C++ interface compatible with Gazebo sensor plugins.

Built to validate the architecture needed for a vendor-agnostic sonar
implementation before writing a GSoC proposal.

---

## Architecture
```
C++ Host / Gazebo Plugin
        │
        │  sonar_engine_init()    ← Load()
        │  sonar_engine_update()  ← Update()
        │  sonar_engine_destroy() ← Unload()
        ▼
Rust FFI Layer (SonarEngine)
        ▼
wgpu Compute Pipeline
        ▼
WGSL Shader — ray marching + wavy terrain + acoustic noise
        ▼
GPU (Vulkan) — Intel / NVIDIA / AMD
```

GPU context created once in `init()`, reused across all `update()` calls.
Same pattern Gazebo sensor plugins use internally.

---

## Project Structure
```
sonar_bridge_poc/
├── src/
│   ├── lib.rs            — SonarEngine struct + C FFI exports
│   └── main.rs           — benchmark runner
├── shaders/
│   └── sonar.wgsl        — WGSL compute shader
├── cpp_host/
│   ├── sonar_engine.h    — C header for plugin integration
│   └── main.cpp          — Gazebo lifecycle simulation
├── gazebo_plugin/
│   └── SonarPlugin.cpp   — plugin stub (no Gazebo install needed)
├── visualize.py          — Python heatmap of sonar output
└── CMakeLists.txt        — ament_cmake compatible build
```

---

## Build and Run
```bash
# Full build + all tests
bash build_and_run.sh

# Or manually:
cargo build --release
cargo run --release
```

---

## Benchmark Results

Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU, Vulkan backend
```
GPU init (one-time, Gazebo Load()):  2.16s

Dispatch times per Update() call:
  256 rays   →  0.29 ms
  1024 rays  →  0.11 ms
  4096 rays  →  0.10 ms
  16384 rays →  0.14 ms
  65536 rays →  0.86 ms
  262144 rays → 1.33 ms
```

At 1024 rays, steady-state cost is ~0.085ms.
Gazebo runs at 60Hz = 16.6ms per frame.
Sonar compute uses under 1% of the frame budget.

![Sonar heatmap and benchmark](sonar_demo.png)

---

## C++ Lifecycle Test
```bash
g++ -std=c++17 -O2 -o gazebo_plugin_test \
    gazebo_plugin/SonarPlugin.cpp \
    -L./target/release -lsonar_engine \
    -Wl,-rpath,./target/release -ldl -lpthread -lm

./gazebo_plugin_test
```

Output:
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

---

## What the shader does

Each GPU thread handles one sonar ray.
Ray steps forward from origin at 5m height.
Floor is a sine-wave terrain (not flat) — different rays return different distances.
Acoustic backscatter noise added per ray.

rays[0]   = 23.97m  ← hit wavy terrain at ~24m
rays[511] = 50.00m  ← reached max range, no hit
rays[1023]= 50.01m  ← reached max range + noise

---

## Purpose

Testing ground for:
- wgpu compute pipeline in a headless (no window) configuration
- Rust to C++ FFI with stable C ABI
- persistent GPU buffer management
- integration patterns for Gazebo sensor plugins

---

## License
MIT
EOF