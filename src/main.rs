// Discrete Event Model
// #[allow(dead_code, unused_imports, unused_variables)]
mod pq;
mod sim;
mod tx;

fn main() {
    run();
}

fn run() {
    let working_dir = "./tmp/sim/test-50yr";

    // Survival curve
    let probabilities = vec![
        1.0, 0.99, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.88, 0.85, 0.82, 0.8, 0.75, 0.7,
        0.65, 0.6, 0.55, 0.5, 0.4, 0.25, 0.15, 0.1, 0.0,
    ];

    let init_states = pq::read("data/init_states.parquet").expect("failted to read init states...");

    let uuids = tx::col_to_vec_str(&init_states, "uuid");
    let states = tx::col_to_vec_i64(&init_states, "step_0");
    let costs = tx::col_to_vec_i64(&init_states, "value");

    // Execute
    let n_steps: i64 = 50;
    let n_sims: i64 = 200;
    let constraints = vec![200_000_000; n_steps as usize];

    if !std::path::Path::new(&working_dir).exists() {
        std::fs::create_dir(&working_dir).expect("failed to create working dir...");
    }

    sim::engine(
        &working_dir,
        n_sims,
        n_steps,
        uuids,
        states,
        probabilities,
        0,
        Some(&costs),
        Some(constraints),
    );

    // Check
    println!("Result Sample :");

    let costs = pq::read(&format!("{}/costs.parquet", &working_dir))
        .expect("failed to read parquet dir...");

    println!("{:?}", costs);
    println!();

    let costs_constrained = pq::read(&format!("{}/costs_const.parquet", &working_dir))
        .expect("failed to read parquet dir...");

    println!("{:?}", costs_constrained);
    println!();
}
