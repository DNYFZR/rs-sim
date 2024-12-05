// Discrete Event Simulator
use ndarray_rand::rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct DiscreteEvent {
  pub states: Vec<i64>,
  pub probabilities: Vec<f64>,
  pub state_matrix: Vec<Vec<Vec<i64>>>,
  pub event_matrix: Vec<Vec<Vec<i32>>>,
}

impl DiscreteEvent {
    fn get_state_matrix(&mut self, n_steps: i64, n_sims: i64) {
        // Initialise random value stream
        let thrd = SmallRng::from_entropy();

        // Setup iter params
        let iter_batch = 64;
        let full_sims = n_sims / iter_batch;
        let last_sim = n_sims - (iter_batch * full_sims);

        // Generate state matrix
        self.state_matrix = Vec::with_capacity(n_sims as usize);
        for _ in 0..full_sims {
            let sim_batch: Vec<Vec<Vec<i64>>> = (0..iter_batch).into_par_iter().map(|_| {
                let mut thrd_para = thrd.clone();
                let mut table = Vec::with_capacity(self.states.len());
                for val in self.states.iter() {
                    // Init val at prev step (iter starts at 1 below)
                    let mut active_value = *val;
    
                    // Expand rows
                    let mut row = Vec::with_capacity(n_steps as usize);
                    row.push(*val);
                    for _ in 1..n_steps {
                        let new_val = active_value + 1;
                        if let Some(prob) = self.probabilities.get(new_val as usize) {
                            if prob > &thrd_para.gen() {
                                active_value = new_val;
                            } else {
                                active_value = 0;
                            }
                        } else {
                            active_value = 0;
                        }
    
                        row.push(active_value);
                    }
    
                    table.push(row);
                }
    
                table
            }).collect();
            self.state_matrix.extend(sim_batch);
        }

        if last_sim > 0 {
            let sim_batch: Vec<Vec<Vec<i64>>> = (0..last_sim).into_par_iter().map(|_| {
                let mut thrd_para = thrd.clone();
                let mut table = Vec::with_capacity(self.states.len());
                for val in self.states.iter() {
                    // Init val at prev step (iter starts at 1 below)
                    let mut active_value = *val;
    
                    // Expand rows
                    let mut row = Vec::with_capacity(n_steps as usize);
                    row.push(*val);
                    for _ in 1..n_steps {
                        let new_val = active_value + 1;
                        if let Some(prob) = self.probabilities.get(new_val as usize) {
                            if prob > &thrd_para.gen() {
                                active_value = new_val;
                            } else {
                                active_value = 0;
                            }
                        } else {
                            active_value = 0;
                        }
    
                        row.push(active_value);
                    }
    
                    table.push(row);
                }
    
                table
            }).collect();
            self.state_matrix.extend(sim_batch);
        }
        
    }

    fn get_event_matrix(&mut self, target_value:i64) {
      // Convert such that target value = 1 & all others = 0
      self.event_matrix = self.state_matrix.iter().map(|table| {
        return table.iter().map(|row: &Vec<i64>| {
          return row.iter().map(|val: &i64| {
              if val == &target_value {return 1} 
              else { return 0} 
          }).collect::<Vec<i32>>();
      }).collect::<Vec<Vec<i32>>>();
      }).collect::<Vec<Vec<Vec<i32>>>>();
    }

    pub fn aggregate(&mut self) -> Vec<Vec<i32>> {
        let agg = self.event_matrix
            .iter()
            .map(|table| {
                let mut table_res: Vec<i32> = vec![];
                let _tmp = table.clone().iter_mut().enumerate().map(|(idx, row)| {   
                    if idx == 0 {
                        table_res = row.clone();
                    }

                    else {
                        table_res = table_res.iter().zip(row.clone()).map(|(a, b)| a.clone() + b.clone()).collect();
                    }
                    return row.clone();
                }).collect::<Vec<Vec<i32>>>();
                return table_res;
            }).collect();

        return agg;
    }

}


pub fn run(states:Vec<i64>, probabilities:Vec<f64>, n_steps:i64, n_sims:i64) -> DiscreteEvent {
  
  let mut event = DiscreteEvent{
    states: states.clone(),
    probabilities: probabilities.clone(),
    state_matrix: vec![vec![vec![0]], ],
    event_matrix: vec![vec![vec![0]], ],
  };

  event.get_state_matrix(n_steps, n_sims);
  event.get_event_matrix(0);

  return event;
}

