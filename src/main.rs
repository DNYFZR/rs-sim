#[allow(dead_code, unused_imports, unused_variables)]
mod sim;

use std::time::Instant;
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
    let mut rng = SmallRng::from_entropy();
    let mut states:Vec<i64> = vec![];
    for _ in 0..100_000 {
        states.push(rng.gen_range(0..50));
    }
    let states_len = states.len();

    // Run settings
    let n_steps = 50;
    let n_sims = 100;
    
    // Execute - maxes out at 300 in parallel with 16GB RAM - CPU ok
    println!("Simulation initialised : {:#?} simulations of {:#?}k assets over {:#?} timesteps", n_sims, &states_len/1000, n_steps);
    
    let start = Instant::now();
    sim::execute(n_sims, n_steps, states, probabilities);
    let duration = start.elapsed();
    
    println!("Simulation complete in : {:?}", duration);
    println!();

    // Check result
    let f = sim::read_parquet_file("./tmp/result_0.parquet")
        .expect("failed to read parquet dir...");
    println!("{:?}", f);

}


