<h2 align="center"> rs-sim </h2>

This rust codebase is for running discrete event simulations

### Functionality

- parallelised execution engine, operating within local memory constraints

- core discrete event simulation engine

- event constraint engine - apply a financial constraint to results, reallocating events to prevent breaching each timestep limit

- result aggregation systems
  
  - summarise total events per timestep across all simulations
  
  - the same, but using a converted target value (e.g. replacement cost / asset value etc.)
  
  - profile values across each timestep within each iteration

- file i/o management (parquet)

### To Do

- Implement testing

- In year risk modelling - unconstrained demand vs. constrained 'reality'

- Add UUID handling & review data flows

- Add input file handling (states & probabilities etc.)

- Add UI (TUI or GUI) - TBD
