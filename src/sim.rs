// Simulation Module
use crate::pq;
use crate::tx;

use ndarray_rand::rand::{Rng, SeedableRng, rngs::SmallRng};
use polars::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

pub fn engine(
    output_dir: &str,
    n_sims: i64,
    n_steps: i64,
    uuids: Vec<String>,
    states: Vec<i64>,
    probabilities: Vec<f64>,
    target_value: i64,
    costs: Option<&Vec<i64>>,
    limit_array: Option<Vec<i64>>,
) {
    println!(
        "Simulation initialised : {:#?} simulations of {:#?}k assets over {:#?} timesteps",
        n_sims,
        &states.len() / 1000,
        n_steps
    );
    let start = Instant::now();

    // Configrue parallel loops
    let para_limit = 100;
    let batches = n_sims / para_limit;
    let remainder = n_sims - batches * para_limit;

    let loop_batches = match remainder {
        0 => batches,
        _ => batches + 1,
    };

    // Run
    for batch in 0..loop_batches {
        let batch_size = if n_sims < para_limit {
            n_sims
        } else if batch > batches {
            remainder
        } else {
            para_limit
        };
        let sim_batch_id = batch * para_limit;

        // Run Sim
        (0..batch_size)
            .into_par_iter()
            .map(|sim_id| {
                // Run sim
                let run_id: i64 = sim_batch_id + sim_id;
                execute_event(
                    output_dir,
                    run_id,
                    &uuids,
                    &states,
                    &probabilities,
                    &n_steps,
                    costs,
                    limit_array.clone(),
                )
                .expect(&format!("Execution failed for run ID : {run_id}"));

                // return original value
                return sim_id;
            })
            .collect::<Vec<i64>>();

        println!("Batch {} of {} processed", batch + 1, loop_batches);
    }

    let duration = start.elapsed();
    println!("Simulation complete in : {:?}", duration);
    println!();

    // Aggregations
    if costs.is_some() {
        tx::aggregate(output_dir, n_sims, target_value, true, false)
            .expect("failed to complete aggregation...");

        if limit_array.is_some() {
            tx::aggregate(output_dir, n_sims, target_value, true, true)
                .expect("failed to complete constrained aggregation...");
        }
    } else {
        tx::aggregate(output_dir, n_sims, target_value, false, false)
            .expect("failed to complete aggregation...");
    }

    let duration_agg = start.elapsed();
    println!("Aggregation complete in : {:?}", duration_agg - duration);
    println!();
}

fn execute_event(
    output_dir: &str,
    run_id: i64,
    uuids: &Vec<String>,
    states: &Vec<i64>,
    probabilities: &Vec<f64>,
    n_steps: &i64,
    costs: Option<&Vec<i64>>,
    limit_array: Option<Vec<i64>>,
) -> Result<(), PolarsError> {
    let costs_is_some = costs.is_some();

    let mut tmp_df = vec![Column::new(PlSmallStr::from_str("uuid"), uuids)];
    tmp_df.extend(
        discrete_event(states, probabilities, n_steps)
            .iter()
            .enumerate()
            .map(|(n, c)| Column::new(PlSmallStr::from_str(&format!("step_{n}")), c))
            .collect::<Vec<Column>>(),
    );

    let mut tmp_df = &mut DataFrame::new(tmp_df).expect("failed to create table...");

    if costs_is_some {
        tmp_df = tmp_df.with_column(Series::from_vec(
            PlSmallStr::from_str("cost"),
            costs.unwrap().clone(),
        ))?;
    }

    // Add id column
    tmp_df = tmp_df.with_column(Series::from_vec(
        PlSmallStr::from_str("sim_id"),
        vec![run_id as i64; tmp_df.shape().0],
    ))?;

    // Write to storage
    pq::write(
        tmp_df.clone(),
        &format!("{}/result_{}.parquet", &output_dir, &run_id),
    )?;

    // Run sim constraint - reduce file i/o
    if limit_array.is_some() && costs_is_some {
        let tmp_df_const = tx::constrain_event(tmp_df, limit_array.unwrap())?;

        // Write to storage
        pq::write(
            tmp_df_const,
            &format!("{}/result_{}_constrained.parquet", &output_dir, &run_id),
        )?;
    }

    Ok(())
}

fn discrete_event(states: &Vec<i64>, probabilities: &Vec<f64>, n_steps: &i64) -> Vec<Vec<i64>> {
    // It is 2x faster to transpose the row results than run as columns
    return tx::transpose(
        &states
            .into_par_iter()
            .map(|&v| {
                let mut para_thrd = SmallRng::from_entropy();

                let mut row: Vec<i64> = Vec::with_capacity(*n_steps as usize);
                row.push(v);

                let mut active_value = v;
                for _ in 1..*n_steps {
                    let new_val = active_value + 1;
                    if let Some(prob) = probabilities.get(new_val as usize) {
                        if prob > &para_thrd.r#gen() {
                            active_value = new_val;
                        } else {
                            active_value = 0;
                        }
                    } else {
                        active_value = 0;
                    }

                    row.push(active_value);
                }
                return row;
            })
            .collect::<Vec<Vec<i64>>>(),
    );
}

#[test]
fn test_discrete_event() {
    let probabilities = vec![
        1.0, 0.99, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.88, 0.85, 0.82, 0.8, 0.75, 0.7,
        0.65, 0.6, 0.55, 0.5, 0.4, 0.25, 0.15, 0.1, 0.0,
    ];

    let init_states = pq::read("data/demo_input.parquet").expect("failted to read init states...");

    // let uuids = tx::col_to_vec_str(&init_states, "uuid");
    let states = tx::col_to_vec_i64(&init_states, "step_0");
    // let costs = tx::col_to_vec_i64(&init_states, "value");
    let n_steps: i64 = 5;

    let res = discrete_event(&states, &probabilities, &n_steps);
    assert!(res[0].len() == 100_000);
}

#[test]
fn test_execute_event() {
    let probabilities = vec![
        1.0, 0.99, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.88, 0.85, 0.82, 0.8, 0.75, 0.7,
        0.65, 0.6, 0.55, 0.5, 0.4, 0.25, 0.15, 0.1, 0.0,
    ];

    let init_states = pq::read("data/demo_input.parquet").expect("failted to read init states...");

    let uuids = tx::col_to_vec_str(&init_states, "uuid");
    let states = tx::col_to_vec_i64(&init_states, "step_0");
    // let costs = tx::col_to_vec_i64(&init_states, "value");
    let n_steps: i64 = 5;

    // run base sim only, with event aggregation & value counting (age profiles)
    if !std::fs::exists("./tmp").unwrap() {
        std::fs::create_dir("./tmp").unwrap();
    }
    execute_event(
        "./tmp/",
        0,
        &uuids,
        &states,
        &probabilities,
        &n_steps,
        None,
        None,
    )
    .unwrap();

    let res = pq::read("./tmp/result_0.parquet").unwrap();
    std::fs::remove_file("./tmp/result_0.parquet").unwrap();
    assert!(res.shape().0 == 100_000);
}
