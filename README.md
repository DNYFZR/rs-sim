<h1 align="center"> rs-sim </h1>
<pre align="center">discrete event simulations using rust</pre>

### Functionality

- event engine for parallel execution of models, within local memory constraints

- core discrete event simulator

- parquet file i/o

- aggregation system
  
  - summarise total events per timestep in each simulation & combine into a single table
  
  - the same, but using a converted target value (e.g. replacement cost)
  
  - the age profile across each timestep for each iteration

- apply a financial constraint to the resulting simulation and model across full timelines

- more to follow...

### Dependencies

- **ndarray-rand** : random number generation
- **rayon** : parallel processing engine
- **polars** : dataframe ops & parquet file i/o
