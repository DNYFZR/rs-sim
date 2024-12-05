// #[allow(dead_code, unused_imports, unused_variables)]
mod sim;

use std::time::Instant;
use ndarray_rand::rand::{thread_rng, Rng};

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
    let mut rng = thread_rng();
    let mut states:Vec<i64> = vec![];
    for _ in 0..100_000 {
        states.push(rng.gen_range(0..50));
    }

    // Run
    let n_steps = 50;
    let n_sims = 10;
    println!("Simulation initialised : {:#?} simulations of {:#?}k assets over {:#?} timesteps", n_sims, states.len()/1000, n_steps);

    let start = Instant::now();
    let mut event = sim::run(states, probabilities, n_steps, n_sims);
    let duration = start.elapsed();

    let start = Instant::now();
    let agg = event.aggregate();
    let duration_2 = start.elapsed();

    // Result
    println!("Simulation complete in : {:?}", duration);
    println!("Aggregation complete in : {:?}", duration_2);
    println!();

    println!("Sample :");
    println!("{:?}", &event.state_matrix[0][0]);
    println!("{:?}", &event.event_matrix[0][0]);
    println!("{:?}", &agg[0]);

}
