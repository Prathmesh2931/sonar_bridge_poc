use std::time::Instant;

fn main() {
    
    let ray_counts = [128, 256, 512,  1024, 4096, 8192 , 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
    let mut results = vec![];

    for &count in &ray_counts {
        let mut rays: Vec<f32> = vec![0.0; count];

        let start   = Instant::now();
        sonar_engine::process_rays(&mut rays);
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0;
        println!("Rays: {:>5} | Time: {:>7.2} ms | Sample: rays[0]={:.4}",
            count, ms, rays[0]);

        results.push((count, ms));
    }

    // ── Print benchmark table ─────────────────────────────────────────────────
    println!("\n┌─────────────┬────────────────┐");
    println!("│  Ray Count  │   Time (ms)    │");
    println!("├─────────────┼────────────────┤");
    for (count, ms) in &results {
        println!("│ {:>11} │ {:>14.2} │", count, ms);
    }
    println!("└─────────────┴────────────────┘");
}
