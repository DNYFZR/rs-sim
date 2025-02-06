// File I/O
use polars::{prelude::*, io::parquet::write::StatisticsOptions};

pub fn read(path:&str) -> PolarsResult<DataFrame> {
  Ok(
      ParquetReader::new(std::fs::File::open(path)?)
          .use_statistics(true)
          .finish()?
  )
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
