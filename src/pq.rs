// File I/O
use polars::{prelude::*, io::parquet::write::StatisticsOptions};

pub fn read(path:&str) -> PolarsResult<DataFrame> {
  Ok(
      ParquetReader::new(std::fs::File::open(path)?)
          .use_statistics(true)
          .finish()?
  )
}

#[test]
fn test_read(){
    // Ensure expected columns are in table 
    let test = vec!["uuid", "value", "step_0"];
    let res:Vec<String> = read("./data/init_states.parquet")
        .expect("failed to read file")
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    assert_eq!(test, res);
}

pub fn write(mut df: DataFrame, path: &str) -> Result<(), PolarsError> {
    let file = std::fs::File::create(path)?;

    // Create a ParquetWriter and write the DataFrame
    ParquetWriter::new(file)
        .with_statistics(StatisticsOptions::full())
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)?;

Ok(())
}

#[test]
fn test_write(){
    // Create file
    if !std::fs::exists("./tmp").unwrap() {
        std::fs::create_dir("./tmp").unwrap();
    }
    let test = "./tmp/test_write.parquet";
    let df = read("./data/init_states.parquet").expect("failed to read file");
    write(df, test).expect("failed to write...");

    // Confirms existance & remove
    let res = std::fs::exists(test).unwrap();
    std::fs::remove_file(test).unwrap();

assert!(res);

}
