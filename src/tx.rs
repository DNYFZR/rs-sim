// Transformation Module
use crate::pq;
use rayon::prelude::*;
use polars::prelude::*;
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};

pub fn aggregate(output_dir:&str, n_sims:i64, target_value:i64, target_mapping:Option<&Vec<i64>>, constrained_scenarios:bool) -> Result<(), PolarsError> {       
  // Create result summaries
  let mut profiles = vec![];
  let mut agg_counts = vec![];
  let mut agg_values = vec![];
  
  for i in 0..n_sims {
      let path = if constrained_scenarios {&format!("{}/result_{}_constrained.parquet", &output_dir, i)} else {&format!("{}/result_{}.parquet", &output_dir, i)};
      let table = pq::read(&path)?;
      let map = vec![1 as i64; table.shape().0];
  
      profiles.push(count_values(&table).lazy());
      let converted = convert(table, target_value, map);
      let aggregated = aggregate_event(converted ).lazy(); 
     
      if target_mapping.is_some() {
        agg_values.push(aggregated.clone());
      }

      agg_counts.push(aggregated);
  }

  // Consolidate arrays
  let agg_count = concat(agg_counts, UnionArgs::default())?.collect()?;
  let path = if constrained_scenarios {&format!("{}/events_const.parquet", &output_dir)} else {&format!("{}/events.parquet", &output_dir)};
  pq::write(agg_count, &path)?;

  let agg_value = concat(agg_values, UnionArgs::default())?.collect()?;  
  let path = if constrained_scenarios {&format!("{}/costs_const.parquet", &output_dir)} else {&format!("{}/costs.parquet", &output_dir)};
  pq::write(agg_value, &path)?;

  let profile = concat(profiles, UnionArgs::default())?.collect()?;
  let path = if constrained_scenarios {&format!("{}/profile_const.parquet", &output_dir)} else {&format!("{}/profile.parquet", &output_dir)};
  pq::write(profile, &path)?;

  Ok(())
}

fn aggregate_event(table:DataFrame) -> DataFrame {
  // Get non-id cols
  let agg_cols:Vec<String> = table.get_column_names()
      .into_par_iter()
      .filter(|s| s.to_string().contains("step"))
      .map(|v| v.to_string())
      .collect();

  
  return table.lazy()
      .drop(["cost"])
      .group_by(["sim_id"])
      .agg([cols(&agg_cols).sum()])
      .collect()
      .expect("failed to aggregate...");
}

pub fn convert(table:DataFrame, target_value:i64, target_mapping: Vec<i64>) -> DataFrame {
  // Get non-id cols
  let agg_cols:Vec<String> = table.get_column_names()
      .into_par_iter()
      .filter(|s| s.to_string() != PlSmallStr::from_str("sim_id").to_string())
      .map(|v| v.to_string())
      .collect();

  // Create target map column
  let target_map_col = Series::from_vec(PlSmallStr::from_str("cost"), target_mapping);
  
  // Run conversion
  return table.lazy()
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

pub fn count_values(table:&DataFrame) -> DataFrame {
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
            //   .collect()
            //   .expect("failed to add sim ID...");
      }).collect::<Vec<LazyFrame>>();
  
  // update table
  let mut df = container[0].clone();
  for idx in 1..container.len() {
    if idx == 0 {
          continue;
      } else {
          df = df.lazy().join(
            container[idx].clone(), 
              [col("sim_id"), col("value")], 
              [col("sim_id"), col("value")],
              JoinArgs::new(JoinType::Left),
          );
      }
  }
      
  // Replace nulls with zero
  return df
      .fill_null(0)
      .sort(["value"], SortMultipleOptions::default())
      .collect().expect("failed to sort...");
}

pub fn constrain_event(mut table:DataFrame, limit_array:Vec<i64>) -> Result<DataFrame, PolarsError> {
  let val_col = "cost";
  let step_col = "step";
  let mut thrd = SmallRng::from_entropy();
  let active_cols:Vec<(usize, String)> = table.get_column_names()
      .iter()
      .enumerate()
      .filter(|(_, c)| c.contains(step_col))
      .map(|(i,c)| (i, c.to_string()))
      .collect();

  for (idx, col_name) in &active_cols {
      if *idx > 0 {
          let ord_col = Series::from_vec(
              PlSmallStr::from_str("order_col"), 
              (0..table.shape().0).map(|_| thrd.gen::<f64>()).collect::<Vec<f64>>() 
          );

          // Add order & applied costs
          table = table.lazy().with_columns([
              lit(ord_col).alias("order_col"),
          ]).collect()?;

          table = table.lazy().with_columns([
              when(col(col_name).eq(lit(0))).then(col(val_col)).otherwise(lit(0)).alias("applied_cost"),
          ]).collect()?;

          // Sort & totalise
          table = table.sort(["order_col"], SortMultipleOptions::new().with_order_descending(true))
              .expect("failed to sort...")
              .lazy().with_columns([
                  col("applied_cost").cum_sum(false).alias("totaliser"),
              ]).collect()?;

          // Update working col
          table = table.lazy().with_columns([lit(limit_array[*idx]).alias("set_limit")]).collect()?;

          table = table.lazy().with_columns([
              when(col(col_name).eq(lit(0)).and(col("totaliser").gt(col("set_limit")).eq(lit(true))) )
              .then(col(&active_cols[idx-1].1) + lit(1))
              .otherwise(col(col_name))
              .alias("tmp_col"),
          ]).collect()?;

          for (i, c) in active_cols.iter().rev() {
              if i > &idx {
                  table = table.lazy().with_column(
                      when(col(col_name).ne(&col("tmp_col")))
                      .then(col(&active_cols[i - 1].1))
                      .otherwise(col(c))
                  ).collect()?;
              }
          }

          table = table.lazy().with_column(col("tmp_col").alias(col_name)).collect()?;

      } else {
          continue;
      }
  }
  table = table.drop_many(["order_col", "applied_cost", "totaliser", "tmp_col", "set_limit"]);

  return Ok(table);
}

pub fn transpose(v:&Vec<Vec<i64>>) -> Vec<Vec<i64>> {
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

