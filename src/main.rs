mod physics_engine;
use std::time::Instant;
use crate::physics_engine::{SonarConfig, SonarPhysicsEngine, PhysicsInput};

fn main() {
    println!("=== sonar_bridge_poc — physics pipeline ===\n");

    let config = SonarConfig {
        n_beams:     512,
        n_rays:      114,
        n_freq:      512,
        sound_speed: 1500.0,
        bandwidth:   2_950.0,
        max_range:   60.0,
        attenuation: 0.0,
        h_fov:       std::f32::consts::PI * 90.0 / 180.0,
        v_fov:       std::f32::consts::PI * 20.0  / 180.0,
        mu_default:  0.5,
    };

    let engine = SonarPhysicsEngine::new(config.clone());
    println!("GPU: {}", engine.gpu_name());

    let n_total = (config.n_beams * config.n_rays) as usize;
    let depth   = vec![10.0f32; n_total];
    let mut normals = vec![0.0f32; n_total * 3];
    for i in 0..n_total { normals[i * 3 + 1] = 1.0; }
    let refl = vec![0.5f32; n_total];
    let mut beam_corr = vec![0.0f32; (config.n_beams * config.n_beams) as usize];
    for i in 0..config.n_beams as usize {
        beam_corr[i * config.n_beams as usize + i] = 1.0;
    }

    let input = PhysicsInput {
        depth: &depth, normals: &normals, reflectivity: &refl,
        beam_corrector: &beam_corr,
        beam_corr_sum: config.n_beams as f32,
        frame: 0, seed: 42,
    };

    println!("\n--- Stage 2: physics pipeline benchmark ---");
    println!("  warmup (shader JIT)...");
    let warmup = engine.run(&input);
    println!("  warmup: {:.2} ms", warmup.compute_ms);

    let n_runs = 10u32;
    let t = Instant::now();
    for frame in 1..=n_runs {
        let inp = PhysicsInput { frame, ..input };
        engine.run(&inp);
    }
    let avg_ms = t.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;
    println!("  avg over {} runs: {:.3} ms/frame", n_runs, avg_ms);
    println!("  @ 10 Hz budget (100ms): {:.1}% used", avg_ms);

    println!("\n--- Stage 3: physical validation ---");
    let out = engine.run(&input);

    let range_res  = config.sound_speed / (2.0 * config.bandwidth);
    let r_floor    = 10.0_f32;
    let expect_bin = (r_floor / range_res) as usize % config.n_freq as usize;
    let peak_bin   = out.peak_bin();
    let diff = (peak_bin as i64 - expect_bin as i64).unsigned_abs() as usize;

    println!("  range resolution : {:.6} m/bin", range_res);
    println!("  expected bin     : {}  (10m floor, r = t·c/2B)", expect_bin);
    println!("  observed peak    : {}", peak_bin);
    if diff <= 1 {
        println!("  ✓ PASS (diff={})", diff);
    } else {
        println!("  ✗ FAIL (diff={})", diff);
    }

    println!("\n  beam 0 — first 10 range bins:");
    for f in 0..10usize {
        println!("    bin {:>3}  r={:.4}m  I={:.4e}",
            f, f as f32 * range_res, out.intensity[f]);
    }

    println!("\n  pipeline summary:");
    println!("    Pass 1 (Eq.14+Eq.8): {}×{} rays → {}×{} spectrum",
        config.n_beams, config.n_rays, config.n_beams, config.n_freq);
    println!("    Pass 2 (matmul W·P): {}×{} · {}×{}",
        config.n_beams, config.n_beams, config.n_beams, config.n_freq);
    println!("    Pass 3 (FFT):        {} beams × {}-pt radix-2",
        config.n_beams, config.n_freq);
    println!("\nDone.");
}
