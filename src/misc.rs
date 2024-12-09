
// pub fn append_parquet(mut df: &mut DataFrame, path: &str) -> Result<(), PolarsError> {
//     // handle append
//     if std::fs::exists(path)? {
//         let src = read_parquet_file(&path)?;
//         let table = src.vstack(&df.select(src.get_column_names_str())?)?;
//         *df = table;
//     } 

//     let file = std::fs::OpenOptions::new()
//         .write(true)
//         .create(true)
//         .truncate(true)
//         .open(path)?;
    
//     // Create a ParquetWriter and write the DataFrame
//     ParquetWriter::new(file)
//         .with_statistics(StatisticsOptions::full())
//         .with_compression(ParquetCompression::Snappy)
//         .finish(&mut df)?;

//     Ok(())
// }

// fn something() {
//    use std::sync::{Arc, Mutex};

//     let profiles = Arc::new(Mutex::new(Vec::new()));

//     (0..n_sims).into_par_iter().map(|i| {
//         let table = read_parquet_file(&format!("{}/result_{}.parquet", &output_dir, i))
//             .expect("failed to read file...");
//         let map = vec![1 as i64; table.shape().0];
    
//         let mut profiles = profiles.lock().unwrap();
//         profiles.push(count_values(&table));
    
//         return i;
//     }).collect::<Vec<i64>>();

    
//     // Consolidate arrays
//     let profiles = profiles.lock().unwrap().to_vec();
//     let mut profile = DataFrame::empty();
//     for i in 0..profiles.len() {
//         let res = profiles.get(i).expect("failed to get table...").clone();
//         if i == 0 {
//             profile = res.clone();
//         } else {
//             profile = profile.vstack(&res)
//                 .expect("failed to stack...");
//         }
//     }
//     write_parquet(&mut profile, &format!("{}/profile.parquet", &output_dir))?;
    
// }

// def polar_constrain(df, annual_limit = 250e6, converter_col="converter", iter_regex = "step"):
//     '''Constrain...'''

//     df = df.with_columns(pl.col(converter_col).cast(pl.Int64).alias(converter_col))
//     iter_cols = [i for i in df.columns if re.search(iter_regex, i)] 
    
//     # Create limit map - if not provided
//     if not isinstance(annual_limit, dict):
//         annual_limit = {i: annual_limit for i in iter_cols}

//     # Apply constraint - skip initial state (step_0)
//     for n, col in enumerate(iter_cols):
//         if n > 0 :
//             # Add random value col for ordering window
//             df = df.with_columns(pl.lit(np.random.random_sample(size=df.shape[0])).alias("order_col"))

//             # Calculate totals across window
//             df = df.with_columns(pl.when(pl.col(col) == 0).then(pl.col(converter_col).cast(pl.Int64)).otherwise(0).alias("applied"))
//             df = df.sort("order_col", descending=True).with_columns(pl.col("applied").cumsum().over("model_iteration").alias("total"))

//             # Create / update temp col
//             df = df.with_columns(pl.when((pl.col(col) == 0) & (pl.col("total") > annual_limit[col])).then(pl.col(iter_cols[n-1]) + pl.lit(1)).otherwise(pl.col(col)).alias("tmp_col"))
            
//             # Update from last col to next col
//             for i, c in enumerate(iter_cols[:n:-1]):
//                 df = df.with_columns(pl.when(pl.col("tmp_col") != pl.col(col)).then(pl.col(iter_cols[-(i+2)])).otherwise(pl.col(c)).alias(c))

//             # Update current col
//             df = df.with_columns(pl.col("tmp_col").alias(col))

//     return df