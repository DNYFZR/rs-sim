<h1 align="center"> rs-sim </h1>

This rust codebase is for running discrete event simulations

### Functionality

- parallelised execution engine, operating within local memory constraints

- core discrete event simulation engine

- file i/o management (parquet)

- result aggregation systems
  
  - summarise total events per timestep across all simulations
  
  - the same, but using a converted target value (e.g. replacement cost / asset value etc.)
  
  - profile values across each timestep within each iteration

- event constraint engine - apply a (financial) constraint to results, reallocating events to prevent breaching the timestep limit


### Dependencies

- **ndarray-rand** : random number management
- **rayon** : parallel processing implementation
- **polars** : dataframe operations & parquet file i/o management

### To Do

1. Implement testing

2. In year risk modelling - unconstrained demand vs. constrained 'reality'
