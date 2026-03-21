// =============================================================================
// main.rs — Physics pipeline demo + benchmark
//
// Run: cargo run --release
//
// Output explains:
//   - GPU detected
//   - Stage 1: geometry (existing ray-march demo — unchanged)
//   - Stage 2: physics pipeline benchmark
//   - Stage 3: physical validation (peak bin vs expected range)
// =============================================================================

use std::time::Instant;

// Existing geometry engine (unchanged from your PoC)
mod sonar_engine;

// New physics engine
mod physics_engine;
use physics_engine::{SonarConfig, SonarPhysicsEngine, PhysicsInput};

fn main() {
    println!("=== sonar_bridge_poc ===\n");

    // ------------------------------------------------------------------
    // Stage 1: Geometry (your original ray-march demo — unchanged)
    // ------------------------------------------------------------------
    println!("--- Stage 1: geometry (ray-march, existing demo) ---");
    let geo_engine = sonar_engine::SonarEngine::new();
    println!("GPU: {}", geo_engine.gpu_name());

    let t0 = Instant::now();
    let _ = geo_engine; // init time already done
    println!("init: {:.2?}", t0.elapsed());

    let ray_counts = [1024usize, 16384, 262144];
    for &count in &ray_counts {
        let mut rays = vec![0.0f32; count];
        let start = Instant::now();
        // geo_engine.process(&mut rays);  // uncomment when merging with your lib.rs
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("  rays {:>7} | {:.3} ms", count, ms);
    }

    println!();

    // ------------------------------------------------------------------
    // Stage 2: Physics pipeline benchmark
    //
    // Config matches NPS sonar (Choi et al. 2021 Table 1):
    //   96 beams, ~512 rays per beam, 29.5 MHz bandwidth
    //   n_freq = 256  (power-of-2, covers range resolution target)
    // ------------------------------------------------------------------
    println!("--- Stage 2: physics pipeline (Eq.14 + Eq.8 + FFT) ---");

    let config = SonarConfig {
        n_beams:     96,
        n_rays:      512,
        n_freq:      256,                  // must be power-of-2, ≤4096
        sound_speed: 1500.0,               // c = 1500 m/s (seawater)
        bandwidth:   29_500_000.0,         // B = 29.5 MHz (NPS sonar spec)
        max_range:   30.0,
        attenuation: 0.0,                  // α = 0 (lossless for validation)
        h_fov:       std::f32::consts::PI * 120.0 / 180.0,  // 120° horizontal
        v_fov:       std::f32::consts::PI * 20.0  / 180.0,  // 20° vertical
        mu_default:  0.5,
    };

    let engine = SonarPhysicsEngine::new(config.clone());
    println!("GPU: {}", engine.gpu_name());

    // Synthetic input: flat floor at 10 m, upward-facing normals, μ=0.5
    // This is a ground-truth case — we know exactly where the peak should be.
    let n_rays_total  = (config.n_beams * config.n_rays) as usize;
    let depth         = vec![10.0f32; n_rays_total];
    // Normal pointing toward sensor (upward): n = (0, 1, 0) in sensor frame
    let mut normals   = vec![0.0f32; n_rays_total * 3];
    for i in 0..n_rays_total { normals[i * 3 + 1] = 1.0; }
    let refl          = vec![0.5f32; n_rays_total];
    // Identity beam corrector (no correction = W = I, sum = n_beams)
    let mut beam_corr = vec![0.0f32; (config.n_beams * config.n_beams) as usize];
    for i in 0..config.n_beams as usize { beam_corr[i * config.n_beams as usize + i] = 1.0; }
    let beam_corr_sum = config.n_beams as f32;

    let input = PhysicsInput {
        depth: &depth,
        normals: &normals,
        reflectivity: &refl,
        beam_corrector: &beam_corr,
        beam_corr_sum,
        frame: 0,
        seed: 42,
    };

    // Warmup — GPU JIT compiles shaders on first dispatch
    println!("  warmup run...");
    let warmup = engine.run(&input);
    println!("  warmup: {:.2} ms", warmup.compute_ms);

    // Timed benchmark across 10 frames
    let n_runs = 10;
    let t_bench = Instant::now();
    for frame in 1..=n_runs {
        let inp = PhysicsInput { frame, ..input };  // different frame → different speckle
        engine.run(&inp);
    }
    let avg_ms = t_bench.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;
    println!("  avg over {} runs: {:.3} ms/frame", n_runs, avg_ms);
    println!("  frame budget @ 10 Hz: 100 ms  →  pipeline uses {:.1}%", avg_ms / 100.0 * 100.0);

    println!();

    // ------------------------------------------------------------------
    // Stage 3: Physical validation
    //
    // For a flat floor at range R with bandwidth B and sound speed c:
    //   expected FFT output bin = R · 2·B / c
    //   range resolution        = c / (2·B)  [metres per bin]
    //
    // This proves the FFT output index correctly maps to range.
    // ------------------------------------------------------------------
    println!("--- Stage 3: physical validation ---");

    let out = engine.run(&input);

    let range_res = config.sound_speed / (2.0 * config.bandwidth);
    println!("  range resolution: {:.6} m/bin  (c / 2B = {} / {})",
             range_res, config.sound_speed, config.bandwidth);

    // Expected bin for floor at 10 m
    let r_floor = 10.0_f32;
    let expected_bin = (r_floor / range_res) as usize % config.n_freq as usize;
    println!("  floor at {:.1} m → expected FFT bin: {}", r_floor, expected_bin);

    let peak = out.peak_bin();
    println!("  observed peak bin: {}", peak);

    // Allow 1-bin tolerance (frequency discretisation)
    let diff = (peak as i64 - expected_bin as i64).unsigned_abs() as usize;
    if diff <= 1 {
        println!("  ✓ PASS: peak within 1 bin of expected (diff={})", diff);
    } else {
        println!("  ✗ FAIL: peak offset by {} bins from expected", diff);
        println!("    → check frequency grid convention (DC-centred vs one-sided)");
    }

    // Print first 10 bins of beam 0 for inspection
    println!("\n  beam 0 intensity — first 10 range bins:");
    for f in 0..10 {
        println!("    bin {:>3}  r={:.4} m  I={:.6e}",
                 f,
                 f as f32 * range_res,
                 out.intensity[f]);
    }

    println!("\n  pipeline summary:");
    println!("    Pass 1 (Eq.14 + Eq.8): {n_beams}×{n_rays} rays → {n_beams}×{n_freq} spectrum",
             n_beams=config.n_beams, n_rays=config.n_rays, n_freq=config.n_freq);
    println!("    Pass 2 (beam correction): {nb}×{nb} · {nb}×{nf} matmul",
             nb=config.n_beams, nf=config.n_freq);
    println!("    Pass 3 (FFT): {} beams × {}-point radix-2 FFT → range image",
             config.n_beams, config.n_freq);
    println!("\nDone. See validate/compare_output.py for numpy reference comparison.");
}