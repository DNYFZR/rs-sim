use std::io::Write;

use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use polars::{prelude::*, io::parquet::write::StatisticsOptions};

pub fn simulate(states:Vec<i64>, probabilities:Vec<f64>, n_steps:i64) -> Vec<Vec<i64>> {
    return states.into_par_iter().map(|v| {
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
    }).collect();
}

pub fn tag_events(table:&Vec<Vec<i64>>, target_value:i64) -> Vec<Vec<i64>> {
    return table.into_par_iter().map(|row|{
        return row.into_par_iter().map(|&v|{
            if v == target_value {
                return 1;
            }
             return 0;
        }).collect();
    }).collect();
}

pub fn transpose(v:&Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let wid = v[0].len();

    (0..wid).into_par_iter()
        .map(|i| v.iter().map(|row| row[i]).collect())
        .collect()
}

// pub fn aggregate(table:&Vec<Vec<i64>>, target_value:i64) -> Vec<i64> {
//     // make para...
//     let agg = table
//             .into_par_iter()
//             .map(|table| {
//                 let mut table_res: Vec<i32> = vec![];
//                 let _tmp = table.clone().iter_mut().enumerate().map(|(idx, row)| {   
//                     if idx == 0 {
//                         table_res = row.clone();
//                     }

//                     else {
//                         table_res = table_res.iter().zip(row.clone()).map(|(a, b)| a.clone() + b.clone()).collect();
//                     }
//                     return row.clone();
//                 }).collect::<Vec<Vec<i32>>>();
//                 return table_res;
//             }).collect();

//         return agg;
// }

// Move to an IO mod
pub fn to_df(table:&Vec<Vec<i64>>) -> DataFrame {
    // Convert each inner vector into a Series
    let series: Vec<Column> = table.into_par_iter().enumerate().map(|(i, vec)| {
        Column::new(PlSmallStr::from_str(&format!("step_{}", i)), vec)
    }).collect();

    return DataFrame::new(series).expect("failed ot create df...")
}

pub fn write_parquet(mut df: &mut DataFrame, path: &str) -> Result<(), PolarsError> {
    let file = std::fs::File::create(path)?;

    // Create a ParquetWriter and write the DataFrame
    ParquetWriter::new(file)
        .with_statistics(StatisticsOptions::full())
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)?;

    Ok(())
}

pub fn append_parquet(mut df: &mut DataFrame, path: &str) -> Result<(), PolarsError> {
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

pub fn concat_dfs(df_array: Vec<LazyFrame>) -> DataFrame {
    return concat(df_array, UnionArgs::default())
        .expect("failed to stack...")
        .collect()
        .expect("failed to collect...");
}

pub fn read_parquet_file(path:&str) -> PolarsResult<DataFrame> {
    Ok(
        ParquetReader::new(std::fs::File::open(path)?)
            .use_statistics(true)
            .finish()?
    )
}

// Merge with top version
fn simulation((output_dir, run_id, states, probabilities, n_steps): (&str, i64, Vec<i64>, Vec<f64>, i64)) -> Result<(), PolarsError> {       
    // Run sim
    let mut tmp_df = &mut to_df(&transpose(
        &simulate(states.clone(), probabilities.clone(), n_steps)
    ));

    // Add sim_id column
    tmp_df = tmp_df.with_column(Column::new(
        PlSmallStr::from_str("sim_id"), 
        &vec![run_id as i64; tmp_df.shape().0]))
    .expect("failed to add sim_id to df");

    // Write to storage
    write_parquet(&mut tmp_df, &format!("{}/result_{}.parquet", output_dir, run_id))
}

// Only in mod pub ??
pub fn execute(n_sims:i32, n_steps:i64, states: Vec<i64>, probabilities:Vec<f64>) {
   // Configrue parallel loops
   let para_limit = 100;
   let batches = n_sims / para_limit;
   let remainder = n_sims - batches * para_limit;

   let loop_batches:i32 = match remainder {
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
           let run_id:i64 = (sim_batch_id + sim_id) as i64;
           
           // Run sim - note tuple input
           simulation(("./tmp", run_id, states.clone(), probabilities.clone(), n_steps))
               .expect(&format!("Execution failed for run ID : {run_id}"));

           // return original value
           return sim_id;
       }).collect::<Vec<i32>>();

       println!("Batch {} of {} processed", batch + 1, loop_batches);    
   }
}
