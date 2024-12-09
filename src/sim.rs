use std::time::Instant;
use rayon::prelude::*;
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};
use polars::{prelude::*, io::parquet::write::StatisticsOptions};

pub fn engine(output_dir:&str, n_sims:i64, n_steps:i64, states: Vec<i64>, probabilities:Vec<f64>, target_value:i64, target_mapping: Option<Vec<i64>>) {
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

        // Use Arc/Mutex vec here to cache para results and write in batches / append to single parquet ?
        
        // Run Sim
        (0..batch_size).into_par_iter().map(|sim_id| {            
            // Run sim
            let run_id:i64 = sim_batch_id + sim_id;
            execute(output_dir, run_id, states.clone(), probabilities.clone(), n_steps)
                .expect(&format!("Execution failed for run ID : {run_id}"));

            // return original value
            return sim_id;
        }).collect::<Vec<i64>>();

        println!("Batch {} of {} processed", batch + 1, loop_batches);    
    }

    let duration = start.elapsed();
    println!("Simulation complete in : {:?}", duration);
    println!();

    // Aggregations
    aggregate(output_dir, n_sims as i64, target_value, target_mapping).expect("failed to complete aggregation...");
    
    let duration_agg = start.elapsed();
    println!("Aggregation complete in : {:?}", duration_agg - duration);
    println!();
 }

// Simulate
fn execute(output_dir:&str, run_id:i64, states:Vec<i64>, probabilities:Vec<f64>, n_steps:i64) -> Result<(), PolarsError> {       
    // Run sim
    let mut tmp_df = &mut to_df(&discrete_event(states, probabilities, n_steps));

    // Add id column
    tmp_df = tmp_df
        .with_column(Column::new(PlSmallStr::from_str("sim_id"), vec![run_id as i64; tmp_df.shape().0]))
        .expect("failed to add sim_id to df");

    // Write to storage
    write_parquet(tmp_df, &format!("{}/result_{}.parquet", &output_dir, &run_id))
    // return Ok(tmp_df.clone());
}

fn discrete_event(states:Vec<i64>, probabilities:Vec<f64>, n_steps:i64) -> Vec<Vec<i64>> {
    // It is 2x faster to transpose the row results than run as columns
    return transpose(&states.into_par_iter().map(|v| {
        let mut para_thrd = SmallRng::from_entropy();

        let mut row = Vec::with_capacity(n_steps as usize);
        row.push(v);
        
        let mut active_value = v;
        for _ in 1..n_steps {
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
    }).collect());
}

fn transpose(v:&Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let wid = v[0].len();

    (0..wid).into_par_iter()
        .map(|i| v.iter().map(|row| row[i]).collect())
        .collect()
}

fn to_df(table:&Vec<Vec<i64>>) -> DataFrame {
    // Convert each inner vector into a Series
    let series: Vec<Column> = table.into_par_iter().enumerate().map(|(i, vec)| {
        Column::new(PlSmallStr::from_str(&format!("step_{}", i)), vec)
    }).collect();

    return DataFrame::new(series).expect("failed ot create df...")
}


// Aggregate
fn aggregate(output_dir:&str, n_sims:i64, target_value:i64, target_mapping:Option<Vec<i64>>) -> Result<(), PolarsError> {       
    // Create result summaries
    let mut profiles = vec![];
    let mut agg_counts = vec![];
    let mut agg_values = vec![];
    
    for i in 0..n_sims {
        let table = read_parquet_file(&format!("{}/result_{}.parquet", &output_dir, i))
            .expect("failed to read file...");
        let map = vec![1 as i64; table.shape().0];
    
        profiles.push(count_values(&table));
        agg_counts.push( aggregate_sim(&convert(&table, target_value.clone(), map)) );
        if target_mapping.is_some() {
            agg_values.push(aggregate_sim(&convert(&table, target_value.clone(), target_mapping.clone().unwrap())));
        }
    }

    // Consolidate arrays
    let mut agg_count = DataFrame::empty();
    for i in 0..agg_counts.len() {
        let res = agg_counts.get(i).expect("failed to get table...").clone();
        if i == 0 {
            agg_count = res;
        } else {
            agg_count = concat([agg_count.lazy(), res.lazy()], UnionArgs::default())
            .expect("failed to concat...")
            .collect()
            .expect("failed to collect...");
        }
    }
    write_parquet(&agg_count, &format!("{}/events.parquet", &output_dir))?;


    let mut agg_value = DataFrame::empty();    
    let agg_vals_len = agg_values.len();
    if agg_vals_len > 0 {
        for i in 0..agg_vals_len {
            let res = agg_values.get(i).expect("failed to get table...").clone();
            if i == 0 {
                agg_value = res;
            } else {
                agg_value = concat([agg_value.lazy(), res.lazy()], UnionArgs::default())
                .expect("failed to concat...")
                .collect()
                .expect("failed to collect...");
            }
        }
        write_parquet(&agg_value, &format!("{}/costs.parquet", &output_dir))?;
    }

    let mut profile = DataFrame::empty();
    for i in 0..profiles.len() {
        let res = profiles.get(i).expect("failed to get table...").clone();
        if i == 0 {
            profile = res;
        } else {
            profile = concat([profile.lazy(), res.lazy()], UnionArgs::default())
                .expect("failed to concat...")
                .collect()
                .expect("failed to collect...");
        }
    }
    write_parquet(&profile, &format!("{}/profile.parquet", &output_dir))?;

    Ok(())
}

fn aggregate_sim(table:&DataFrame) -> DataFrame {
    // Get non-id cols
    let agg_cols:Vec<String> = table.get_column_names()
        .into_par_iter()
        .filter(|s| 
            s.to_string() != PlSmallStr::from_str("sim_id").to_string() 
            && s.to_string() != PlSmallStr::from_str("cost").to_string())
        .map(|v| v.to_string())
        .collect();

    
    return table.clone().lazy()
        .drop(["cost"])
        .group_by(["sim_id"])
        .agg([cols(&agg_cols).sum()])
        .collect()
        .expect("failed to aggregate...");
}

fn convert(table:&DataFrame, target_value:i64, target_mapping: Vec<i64>) -> DataFrame {
    // Get non-id cols
    let agg_cols:Vec<String> = table.get_column_names()
        .into_par_iter()
        .filter(|s| s.to_string() != PlSmallStr::from_str("sim_id").to_string())
        .map(|v| v.to_string())
        .collect();

    // Create target map column
    let target_map_col = Series::from_vec(PlSmallStr::from_str("cost"), target_mapping);
    
    // Run conversion
    return table.clone().lazy()
        .with_column(lit(target_map_col))
        .with_columns(agg_cols.iter().map(|col_name|{
            let e = when(col(col_name).eq(target_value))
            .then(col("cost").alias(col_name))
            .otherwise(lit(0 as i64).alias(col_name));

            return e;
        }).collect::<Vec<Expr>>())
        .collect()
        .expect("failed to convert...");
}

fn count_values(table:&DataFrame) -> DataFrame {
    let sim_id:i64 = table.column("sim_id").unwrap().get(0).unwrap().try_extract().unwrap();
    let container = table.get_column_names_str()
        .into_par_iter()
        .filter(|c| *c != "sim_id" && *c != "cost")
        .map(|c|{
            // count values
            let mut tmp_df = table
                .column(&c).unwrap()
                .as_series().unwrap()
                .value_counts(false, false, PlSmallStr::from_str("count"), false).unwrap();
            
            tmp_df.rename(c, PlSmallStr::from_str("value")).expect("failed to rename columns...");
            tmp_df.rename("count", PlSmallStr::from_str(c)).expect("failed to rename columns...");
            
            // add sim_id
            return tmp_df.lazy()
                .with_column(lit(sim_id).alias("sim_id"))
                .select([col("sim_id"), col("value"), col(c)])
                .collect()
                .expect("failed to add sim ID...");
        }).collect::<Vec<DataFrame>>();
    
    // update table
    let mut df = DataFrame::empty().lazy();
    for (idx, chunk) in container.iter().enumerate() {
        if idx == 0 {
            df = chunk.clone().lazy();
        } else {
            df = df.lazy().join(
                chunk.clone().lazy(), 
                [col("sim_id"), col("value")], 
                [col("sim_id"), col("value")],
                JoinArgs::new(JoinType::Left),
            )
        }
    }
        
    // Replace nulls with zero
    return df
        .fill_null(0)
        .sort(["value"], SortMultipleOptions::default())
        .collect().expect("failed to sort...");
}


// File system
pub fn read_parquet_file(path:&str) -> PolarsResult<DataFrame> {
    Ok(
        ParquetReader::new(std::fs::File::open(path)?)
            .use_statistics(true)
            .finish()?
    )
}

pub fn write_parquet(df: &DataFrame, path: &str) -> Result<(), PolarsError> {
    let file = std::fs::File::create(path)?;

    // Create a ParquetWriter and write the DataFrame
    ParquetWriter::new(file)
        .with_statistics(StatisticsOptions::full())
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df.clone())?;

    Ok(())
}
