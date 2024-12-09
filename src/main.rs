// Discrete Event Model
// #[allow(dead_code, unused_imports, unused_variables)]
mod sim;
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};

fn main() {
    // Survival curve
    let probabilities = vec![
        1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 
        0.93, 0.92, 0.92, 0.91, 0.91, 0.9, 0.9, 
        0.88, 0.85, 0.82, 0.8, 0.75, 0.72, 0.7, 
        0.68, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 
        0.38, 0.25, 0.15, 0.12, 0.1, 0.05, 0.0
    ];

    // Initial states
    let n_states:i64 = 100_000;
    let mut rng = SmallRng::from_entropy();
    let mut states:Vec<i64> = vec![];
    for _ in 0..n_states {
        states.push(rng.gen_range(0..50));
    }

    // Execute
    let n_steps = 50;
    let n_sims = 1000;
    let working_dir = "./tmp/test-1000";

    if !std::path::Path::new(&working_dir).exists() {
        std::fs::create_dir(&working_dir).expect("failed to create working dir...");
    }

    sim::engine(&working_dir, n_sims, n_steps, states.clone(), probabilities, 0, Some(vec![10_000; n_states as usize]));

    // Check results
    println!("Result Sample :");
    let events = sim::read_parquet_file(&format!("{}/events.parquet", &working_dir))
        .expect("failed to read parquet dir...");
    println!("{:?}", events);
    println!();
    
    let costs = sim::read_parquet_file(&format!("{}/costs.parquet", &working_dir))
        .expect("failed to read parquet dir...");
    println!("{:?}", costs);
    println!();

    let profiles = sim::read_parquet_file(&format!("{}/profile.parquet", &working_dir))
        .expect("failed to read parquet dir...");
    println!("{:?}", profiles);
    println!();
}


