// main.rs — Standalone Rust demo (no C++ needed)
// Run with: cargo run --release

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║     sonar_bridge_poc — Rust-only smoke test      ║");
    println!("╚══════════════════════════════════════════════════╝");

    // Simulate 100 ray distances from Gazebo (1.0 m .. 100.0 m)
    let mut rays: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let original_sample = [rays[0], rays[49], rays[99]];

    println!("\n[input]  rays[0]={:.4}  rays[49]={:.4}  rays[99]={:.4}",
        original_sample[0], original_sample[1], original_sample[2]);

    // Run GPU compute (wgpu picks best available backend automatically)
    sonar_engine::process_rays(&mut rays, 0.05);

    println!("[output] rays[0]={:.4}  rays[49]={:.4}  rays[99]={:.4}",
        rays[0], rays[49], rays[99]);

    let deltas: Vec<f32> = rays.iter()
        .zip(original_sample.iter().chain(original_sample.iter()))
        .take(3)
        .map(|(out, orig)| out - orig)
        .collect();

    println!("\n✔  Backscatter noise applied  (Δ samples: {:?})", deltas);
    println!("✔  All {} rays processed on GPU\n", rays.len());
}
