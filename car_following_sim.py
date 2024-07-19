import matplotlib.pyplot as plt
from car_following.idm import read_idm_ode_parameters,  idm_initial_conditions, solve_idm, plot_idm_results,find_accelerations
from utils.diff_eq import simulation_params
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point 

v0_lead, v0_follow, T_base, s0, a_base, b, delta, n_simulations, n_cars,distance_vehicle,t_min,t_max,t_step = read_idm_ode_parameters()
t_span,t_eval,num_step = simulation_params(t_min,t_max,t_step)
x0 = idm_initial_conditions(n_cars,distance_vehicle,v0_lead)
results_idm = []
for stochastic in [False,True]:
    idm_ode_args = (n_cars,s0,delta,v0_lead, v0_follow, b, T_base, a_base,stochastic)                
    results = solve_idm(n_simulations,t_span,t_eval, x0, idm_ode_args)
    results = find_accelerations(results)
    results_df = pd.concat([car.to_dataframe() for car in results], axis=0)
    results_idm.append(results_df)
results_df = pd.concat(results_idm, axis=0)
results_df.to_csv('./car_following/car_following_results.csv')
plot_idm_results(results_df)




