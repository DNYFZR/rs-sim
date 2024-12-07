use rayon::prelude::*;
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};
use polars::{prelude::*, io::parquet::write::StatisticsOptions};

// Simulation
pub fn engine(output_dir:&str, n_sims:i64, n_steps:i64, states: Vec<i64>, probabilities:Vec<f64>) {
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
        
        // Parrallel inner loop
        (0..batch_size).into_par_iter().map(|sim_id| {
        
            // Create sim uid
            let run_id:i64 = sim_batch_id + sim_id;
            
            // Run sim - note tuple input
            execute("./tmp", run_id, states.clone(), probabilities.clone(), n_steps)
                .expect(&format!("Execution failed for run ID : {run_id}"));

            // return original value
            return sim_id;
        }).collect::<Vec<i64>>();
 
        println!("Batch {} of {} processed", batch + 1, loop_batches);    
    }

    aggregate(output_dir, n_sims as i64).expect("failed to complete aggregation...");
 }

fn execute(output_dir:&str, run_id:i64, states:Vec<i64>, probabilities:Vec<f64>, n_steps:i64) -> Result<(), PolarsError> {       
    // Run sim
    let mut tmp_df = &mut to_df(&discrete_event(states.clone(), probabilities.clone(), n_steps));

    // Add id column
    tmp_df = tmp_df.with_column(Column::new(
        PlSmallStr::from_str("sim_id"), 
        &vec![run_id as i64; tmp_df.shape().0]))
    .expect("failed to add sim_id to df");

    // Write to storage
    write_parquet(&mut tmp_df, &format!("{}/result_{}.parquet", &output_dir, &run_id))
}

pub fn discrete_event(states:Vec<i64>, probabilities:Vec<f64>, n_steps:i64) -> Vec<Vec<i64>> {
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

pub fn to_df(table:&Vec<Vec<i64>>) -> DataFrame {
    // Convert each inner vector into a Series
    let series: Vec<Column> = table.into_par_iter().enumerate().map(|(i, vec)| {
        Column::new(PlSmallStr::from_str(&format!("step_{}", i)), vec)
    }).collect();

    return DataFrame::new(series).expect("failed ot create df...")
}


// Aggregations
fn aggregate(output_dir:&str, n_sims:i64) -> Result<(), PolarsError> {       
    // Aggregate
    for i in 0..n_sims {
        let mut res = aggregate_sim(
            read_parquet_file(&format!("{}/result_{}.parquet", &output_dir, i))?, 
            0
        );

        // Write to storage
        if i == 0 {
            write_parquet(&mut res, &format!("{}/events.parquet", &output_dir))?
        
        } else {
            append_parquet(&mut res, &format!("{}/events.parquet", &output_dir))?            
        }
    }

    Ok(())
}

fn aggregate_sim(table:DataFrame, target_value:i64) -> DataFrame {
    // Get non-id cols
    let agg_cols:Vec<String> = table.get_column_names()
        .into_iter()
        .filter(|s| s.to_string() != PlSmallStr::from_str("sim_id").to_string())
        .map(|v| v.to_string())
        .collect();

    
    return table.lazy()
        .with_columns(agg_cols.iter().map(|col_name|{
            let e = when(col(col_name).eq(target_value))
            .then(lit(1 as i64).alias(col_name))
            .otherwise(lit(0 as i64).alias(col_name));

            return e;
        }).collect::<Vec<Expr>>())
        .group_by(["sim_id"])
        .agg([cols(&agg_cols).sum()])
        .collect()
        .expect("failed to aggregate...");
}


// Constraints
// pub fn constrain(table:Vec<Vec<i64>>, cost_map:Vec<i64>, limit:i64) -> Vec<Vec<i64>> {
//     // let total_cols = table.len();
//     let mut tbl = table.clone();

//     for col in table {
//         let cost:Vec<i64> = col.iter().enumerate().map(|(i, &v)| v * cost_map[i]).collect();
//         let totaliser:Vec<i64> = cost.clone()
//             .into_par_iter()
//             .enumerate()
//             .map(|(i, v)| {
//                 if i == 0 {
//                     v
//                 } else if i == 1 {
//                     v + &cost[0]
//                 } else {
//                     v + &cost[0..(i-1)].iter().sum::<i64>()
//                 }})
//             .collect();
        
//         tbl.push(totaliser);
//     }

//     return tbl;
// }


// IO Operations
pub fn read_parquet_file(path:&str) -> PolarsResult<DataFrame> {
    Ok(
        ParquetReader::new(std::fs::File::open(path)?)
            .use_statistics(true)
            .finish()?
    )
}

fn write_parquet(mut df: &mut DataFrame, path: &str) -> Result<(), PolarsError> {
    let file = std::fs::File::create(path)?;

    // Create a ParquetWriter and write the DataFrame
    ParquetWriter::new(file)
        .with_statistics(StatisticsOptions::full())
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)?;

    Ok(())
}

fn append_parquet(mut df: &mut DataFrame, path: &str) -> Result<(), PolarsError> {
    // handle append
    if std::fs::exists(path)? {
        let src = read_parquet_file(&path)?;
        let table = src.vstack(&df.select(src.get_column_names_str())?)?;
        *df = table;
    } 

    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    // Create a ParquetWriter and write the DataFrame
    ParquetWriter::new(file)
        .with_statistics(StatisticsOptions::full())
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)?;

    Ok(())
}

