<h2 align="center"> rs-sim </h2>

This rust code is for running discrete event simulations :

- read uuids, initial states & costs from parquet file ✔ 

- parallelised execution engine, operating within local memory constraints ✔

- core discrete event simulation engine ✔

- event constraint engine - apply a financial constraint to results, reallocating events to prevent breaching each timestep limit ✔

- result aggregation systems ✔
  
  - summarise total events per timestep across all simulations ✔
  
  - the same, but using a converted target value (e.g. replacement cost / asset value etc.) ✔
  
  - profile values across each timestep within each iteration ✔

- save all data to parquet files ✔

- Implement testing (66%)

  - Set up CI pipeline ✔

  - Setup sim tests (50%)

  - Setup tx tests (50%)

  - Setup pq tests ✔

- In year risk modelling - unconstrained demand vs. constrained 'reality' ✖

- Add TUI / GUI ❔
