use std::time::Instant;

fn main() {

    println!("sonar engine test");

    let init_start = Instant::now();
    let engine = sonar_engine::SonarEngine::new();
    let init_time = init_start.elapsed();

    println!("gpu: {}", engine.gpu_name());
    println!("init: {:.2?}\n", init_time);

    let ray_counts = [256usize, 1024, 4096, 16384, 65536, 262144];

    for &count in &ray_counts {

        let mut rays = vec![0.0f32; count];

        let start = Instant::now();
        engine.process(&mut rays);
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0;

        println!(
            "rays {:>7} | {:>7.2} ms | r0={:.4}",
            count,
            ms,
            rays[0]
        );
    }

    println!("\ninit cost (once): {:.2?}", init_time);
}