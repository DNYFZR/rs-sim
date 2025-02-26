// For creating the demo_input.parquet file in the data dir...
use crate::pq;

pub fn build_test_data(){
  use polars::prelude::*;
  use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};
  
  let n_states:i64 = 100_000;
  let mut rng = SmallRng::from_entropy();

  let states:Vec<i64> = (0..n_states).map(|_| rng.gen_range(0..40)).collect::<Vec<i64>>();
  let costs = (0..n_states).map(|_| rng.gen_range(5_000..15_000)).collect::<Vec<i64>>();
  let uuids = (0..n_states).map(|x| format!("uuid_{x}")).collect::<Vec<String>>();
  
  let cols = vec![
      Column::new(PlSmallStr::from_str("uuid"), uuids), 
      Column::new(PlSmallStr::from_str("step_0"), states), 
      Column::new(PlSmallStr::from_str("value"), costs),
  ];
  
  let df = DataFrame::new(cols).unwrap();
  pq::write(df, "data/demo_input.parquet").unwrap();
}