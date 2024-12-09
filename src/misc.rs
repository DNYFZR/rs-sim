
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