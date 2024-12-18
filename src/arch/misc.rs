fn something() {
   use std::sync::{Arc, Mutex};

    let profiles = Arc::new(Mutex::new(Vec::new()));

    (0..n_sims).into_par_iter().map(|i| {
        let table = read_parquet_file(&format!("{}/result_{}.parquet", &output_dir, i))
            .expect("failed to read file...");
        let map = vec![1 as i64; table.shape().0];
    
        let mut profiles = profiles.lock().unwrap();
        profiles.push(count_values(&table));
    
        return i;
    }).collect::<Vec<i64>>();

    
    // Consolidate arrays
    let profiles = profiles.lock().unwrap().to_vec();
    let mut profile = DataFrame::empty();
    for i in 0..profiles.len() {
        let res = profiles.get(i).expect("failed to get table...").clone();
        if i == 0 {
            profile = res.clone();
        } else {
            profile = profile.vstack(&res)
                .expect("failed to stack...");
        }
    }
    write_parquet(&mut profile, &format!("{}/profile.parquet", &output_dir))?;
    
}
