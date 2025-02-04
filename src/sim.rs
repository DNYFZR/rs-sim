// Simulation Module
use crate::pq;
use crate::tx;
use std::time::Instant;
use rayon::prelude::*;
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};
use polars::prelude::*;

pub fn engine(output_dir:&str, n_sims:i64, n_steps:i64, states: Vec<i64>, probabilities:Vec<f64>, target_value:i64, target_mapping: Option<&Vec<i64>>, limit_array:Option<Vec<i64>>) {
    println!("Simulation initialised : {:#?} simulations of {:#?}k assets over {:#?} timesteps", n_sims, &states.len()/1000, n_steps);
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
        let batch_size = if n_sims < para_limit {n_sims} else if batch > batches {remainder} else {para_limit};
        let sim_batch_id = batch * para_limit;
        
        // Run Sim
        (0..batch_size).into_par_iter().map(|sim_id| {            
            // Run sim
            let run_id:i64 = sim_batch_id + sim_id;
            execute_event(
                output_dir, 
                run_id, 
                &states, 
                &probabilities, 
                &n_steps, 
                target_mapping, 
                limit_array.clone()
            ).expect(&format!("Execution failed for run ID : {run_id}"));

            // return original value
            return sim_id;
        }).collect::<Vec<i64>>();

        println!("Batch {} of {} processed", batch + 1, loop_batches);    
    }

    let duration = start.elapsed();
    println!("Simulation complete in : {:?}", duration);
    println!();

    // Aggregations
    tx::aggregate(output_dir, n_sims, target_value, target_mapping, false).expect("failed to complete aggregation...");
    if limit_array.is_some() && target_mapping.is_some() {
        tx::aggregate(output_dir, n_sims, target_value, target_mapping, true).expect("failed to complete constrained aggregation...");
    }
    
    let duration_agg = start.elapsed();
    println!("Aggregation complete in : {:?}", duration_agg - duration);
    println!();
 }

fn execute_event(output_dir:&str, run_id:i64, states:&Vec<i64>, probabilities:&Vec<f64>, n_steps:&i64, target_mapping: Option<&Vec<i64>>, limit_array:Option<Vec<i64>>) -> Result<(), PolarsError> {
    let mut tmp_df = &mut tx::to_df(&discrete_event(states, probabilities, n_steps));
            
    if target_mapping.is_some() {
        tmp_df = tmp_df.with_column(Series::from_vec(PlSmallStr::from_str("cost"), target_mapping.unwrap().clone()) )?;
    }
    
    // Add id column
    tmp_df = tmp_df.with_column(Series::from_vec(PlSmallStr::from_str("sim_id"), vec![run_id as i64; tmp_df.shape().0]) )?;

    // Write to storage
    pq::write(tmp_df.clone(), &format!("{}/result_{}.parquet", &output_dir, &run_id))?;


    // Run sim constraint - reduce file i/o
    if limit_array.is_some() && target_mapping.is_some() {
        let tmp_df_const = tx::constrain_event(tmp_df.clone(), limit_array.unwrap())?;

        // Write to storage
        pq::write(tmp_df_const, &format!("{}/result_{}_constrained.parquet", &output_dir, &run_id))?;
    }
    
    Ok(())
}

fn discrete_event(states:&Vec<i64>, probabilities:&Vec<f64>, n_steps:&i64) -> Vec<Vec<i64>> {
    // It is 2x faster to transpose the row results than run as columns
    return tx::transpose(&states.into_par_iter().map(|&v| {
        let mut para_thrd = SmallRng::from_entropy();

        let mut row:Vec<i64> = Vec::with_capacity(*n_steps as usize);
        row.push(v);
        
        let mut active_value = v;
        for _ in 1..*n_steps {
            let new_val = active_value + 1;
            if let Some(prob) = probabilities.get(new_val as usize) {
              if prob > &para_thrd.gen() {
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
    }).collect::<Vec<Vec<i64>>>());
}

