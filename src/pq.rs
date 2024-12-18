// File I/O
use polars::{prelude::*, io::parquet::write::StatisticsOptions};

pub fn read(path:&str) -> PolarsResult<DataFrame> {
  Ok(
      ParquetReader::new(std::fs::File::open(path)?)
          .use_statistics(true)
          .finish()?
  )
}

pub fn write(df: &DataFrame, path: &str) -> Result<(), PolarsError> {
  let file = std::fs::File::create(path)?;

  // Create a ParquetWriter and write the DataFrame
  ParquetWriter::new(file)
      .with_statistics(StatisticsOptions::full())
      .with_compression(ParquetCompression::Snappy)
      .finish(&mut df.clone())?;

  Ok(())
}

// pub fn append(mut df: &mut DataFrame, path: &str) -> Result<(), PolarsError> {
//     // handle append
//     if std::fs::exists(path)? {
//         let src = read(&path)?;
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
