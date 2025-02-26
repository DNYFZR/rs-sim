// Transformation Module
use crate::pq;
use ndarray_rand::rand::{Rng, SeedableRng, rngs::SmallRng};
use polars::prelude::*;
use rayon::prelude::*;

pub fn aggregate(
    output_dir: &str,
    n_sims: i64,
    target_value: i64,
    apply_cost: bool,
    constrained_scenarios: bool,
) -> Result<(), PolarsError> {
    // Create result summaries
    let mut profiles = vec![];
    let mut agg_counts = vec![];
    let mut agg_values = vec![];

    for i in 0..n_sims {
        let path = if constrained_scenarios {
            &format!("{}/result_{}_constrained.parquet", &output_dir, i)
        } else {
            &format!("{}/result_{}.parquet", &output_dir, i)
        };
        let table = pq::read(&path)?;

        profiles.push(count_values(&table).lazy());

        if apply_cost {
            let agg_cost = aggregate_event(convert(&table, &target_value, apply_cost));
            agg_values.push(agg_cost.lazy());
        }

        let agg_count = aggregate_event(convert(&table, &target_value, apply_cost)).lazy();
        agg_counts.push(agg_count);
    }

    // Consolidate arrays
    if apply_cost {
        let agg_count = concat(agg_counts, UnionArgs::default())?.collect()?;
        let path = if constrained_scenarios {
            &format!("{}/events_const.parquet", &output_dir)
        } else {
            &format!("{}/events.parquet", &output_dir)
        };
        pq::write(agg_count, &path)?;
    }

    let agg_value = concat(agg_values, UnionArgs::default())?.collect()?;
    let path = if constrained_scenarios {
        &format!("{}/costs_const.parquet", &output_dir)
    } else {
        &format!("{}/costs.parquet", &output_dir)
    };
    pq::write(agg_value, &path)?;

    let profile = concat(profiles, UnionArgs::default())?.collect()?;
    let path = if constrained_scenarios {
        &format!("{}/profile_const.parquet", &output_dir)
    } else {
        &format!("{}/profile.parquet", &output_dir)
    };
    pq::write(profile, &path)?;

    Ok(())
}

fn aggregate_event(table: DataFrame) -> DataFrame {
    // Get non-id cols
    let agg_cols: Vec<String> = table
        .get_column_names()
        .into_par_iter()
        .filter(|s| s.to_string().contains("step"))
        .map(|v| v.to_string())
        .collect();

    return table
        .lazy()
        .drop(["cost"])
        .group_by(["sim_id"])
        .agg([cols(&agg_cols).sum()])
        .collect()
        .expect("failed to aggregate...");
}

fn convert(table: &DataFrame, target_value: &i64, apply_cost: bool) -> DataFrame {
    // Get non-id cols
    let agg_cols: Vec<String> = table
        .get_column_names()
        .into_par_iter()
        .filter(|&s| s.contains("step_"))
        .map(|v| v.to_string())
        .collect();

    // Run conversion
    return table
        .clone()
        .lazy()
        .with_columns(
            agg_cols
                .iter()
                .map(|col_name| {
                    return when(col(col_name).eq(*target_value).and(apply_cost == true))
                        .then(col("cost").alias(col_name))
                        .otherwise(
                            when(col(col_name).eq(*target_value).and(apply_cost == false))
                                .then(lit(1).alias(col_name))
                                .otherwise(lit(0 as i64).alias(col_name)),
                        );
                })
                .collect::<Vec<Expr>>(),
        )
        .collect()
        .expect("failed to convert...");
}

fn count_values(table: &DataFrame) -> DataFrame {
    let sim_id: i64 = table
        .column("sim_id")
        .unwrap()
        .get(0)
        .unwrap()
        .try_extract()
        .unwrap();
    let container = table
        .get_column_names_str()
        .into_par_iter()
        .filter(|c| *c != "sim_id" && *c != "cost" && *c != "uuid")
        .map(|c| {
            let val_counts = table
                .select(vec![c])
                .unwrap()
                .rename(c, PlSmallStr::from_str("value"))
                .unwrap()
                .column("value")
                .unwrap()
                .as_series()
                .unwrap()
                .value_counts(false, false, PlSmallStr::from_str(c), false)
                .expect("failed to count values");

            return val_counts
                .clone()
                .lazy()
                .with_column(lit(sim_id).alias("sim_id"))
                .select([col("sim_id"), col("value"), col(c)]);
        })
        .collect::<Vec<LazyFrame>>();

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
        .collect()
        .expect("failed to sort...");
}

pub fn constrain_event(table: &DataFrame, limit_array: Vec<i64>) -> Result<DataFrame, PolarsError> {
    let mut table = table.clone();
    let val_col = "cost";
    let step_col = "step";
    let mut thrd = SmallRng::from_entropy();

    let n_rows = table.shape().0;
    let active_cols: Vec<(usize, String)> = table
        .get_column_names()
        .iter()
        // We skip step 0 as the initial state cannot be altered
        .filter(|c| c.contains(step_col) && ***c != PlSmallStr::from_str("step_0"))
        .enumerate()
        .map(|(i, c)| (i + 1, c.to_string()))
        .collect();

    for (idx, col_name) in &active_cols {
        // Add columns for random ordering & applied costs
        table = table
            .lazy()
            .with_columns([
                lit(Series::from_vec(
                    PlSmallStr::from_str("order_col"),
                    (0..n_rows)
                        .map(|_| thrd.r#gen::<f64>())
                        .collect::<Vec<f64>>(),
                ))
                .alias("order_col"),
                when(col(col_name).eq(lit(0)))
                    .then(col(val_col))
                    .otherwise(lit(0))
                    .alias("applied_cost"),
            ])
            .collect()?;

        // Sort & totalise
        table = table
            .sort(
                ["order_col"],
                SortMultipleOptions::new().with_order_descending(true),
            )?
            .lazy()
            .with_columns([col("applied_cost").cum_sum(true).alias("totaliser")])
            .collect()?;

        // Increase age if cost is above set limit
        let step_limit = lit(limit_array[*idx]);
        table = table
            .lazy()
            .with_columns([when(
                col(col_name)
                    .eq(lit(0))
                    .and(col("totaliser").gt(step_limit)),
            )
            .then(col(&format!("step_{}", idx - 1)) + lit(1))
            .otherwise(col(col_name))
            .alias("tmp_col")])
            .collect()?;

        // Update following years when event is deferred
        for (i, c) in active_cols.iter().rev() {
            if i > idx {
                table = table
                    .lazy()
                    .with_columns([when(col("tmp_col").neq(col(col_name)))
                        .then(col(&format!("step_{}", i - 1)))
                        .otherwise(col(c))
                        .alias(c)])
                    .collect()?;
            }
        }

        // Update current year
        table = table
            .lazy()
            .with_columns([col("tmp_col").alias(col_name)])
            .collect()?;

        // Remove process cols
        table = table.drop_many(["order_col", "applied_cost", "totaliser", "tmp_col"]);
    }
    return Ok(table);
}

pub fn transpose(v: &Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let wid = v[0].len();

    (0..wid)
        .into_par_iter()
        .map(|i| v.iter().map(|row| row[i]).collect())
        .collect()
}

pub fn col_to_vec_i64(df: &DataFrame, col: &str) -> Vec<i64> {
    return df
        .column(col)
        .expect("failedto get col...")
        .i64()
        .expect("failed to get i64 array")
        .into_iter()
        .map(|v| v.unwrap())
        .collect();
}

pub fn col_to_vec_str(df: &DataFrame, col: &str) -> Vec<String> {
    return df
        .column(col)
        .expect("failedto get col...")
        .str()
        .expect("failed to get i64 array")
        .into_iter()
        .map(|v| String::from(v.unwrap()))
        .collect();
}

#[test]
fn test_transpose() {
    let test: Vec<Vec<i64>> = vec![vec![1], vec![0], vec![1]];
    let res = transpose(&vec![vec![1, 0, 1]]);

    assert_eq!(test, res);
}

#[test]
fn test_col_to_vec_i64() {
    let table = pq::read("./data/demo_input.parquet").unwrap();
    let res = col_to_vec_i64(&table, "step_0");

    assert!(res.len() == 100_000);
}

#[test]
fn test_col_to_vec_str() {
    let table = pq::read("./data/demo_input.parquet").unwrap();
    let res = col_to_vec_str(&table, "uuid");

    assert!(res.len() == 100_000);
}
