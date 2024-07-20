import matplotlib.pyplot as plt
from car_following.idm import read_idm_ode_parameters,  idm_initial_conditions, solve_idm, plot_idm_results,find_accelerations
from utils.diff_eq import simulation_params
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point 
import pymc as pm
# import pytensor as pt
from scipy.integrate import odeint,solve_ivp
from car_following.idm import idm_ode
import arviz as az
import pandas as pd

if __name__ == "__main__":
    
    observed_data = pd.read_csv('./car_following/car_following_results.csv')
    observed_data=observed_data[(observed_data['stochastic']==False)&(observed_data['simulation_id']==0)]
    observed_data[['position','velocity']] =  np.random.normal(observed_data[['position','velocity']],2)
    observed_data = observed_data[['position','velocity','car_id']]
    observed_data['car_id'] =observed_data['car_id'].astype(int)

    v0_lead, v0_follow, T_base, s0, a_base, b, delta, n_simulations, n_cars,distance_vehicle,t_min,t_max,t_step = read_idm_ode_parameters()
    t_span,t_eval,num_step = simulation_params(t_min,t_max,t_step)
    x0 = idm_initial_conditions(n_cars,distance_vehicle,v0_lead)
    
    results_idm = []
    stochastic = False
    idm_ode_args = (n_cars,s0,delta,v0_lead, v0_follow, b, T_base, a_base,stochastic)
    n_simulations = 1
    T_factors, a_factors = T_base,a_base
    idm_ode_args = idm_ode_args + (T_factors, a_factors)
    sol = solve_ivp(idm_ode, t_span, x0, t_eval=t_eval, vectorized=True, args=(idm_ode_args))
    x0 = x0[0:4]
    data = np.zeros(sol.y.shape)

    def acceleration(v, s,s0,delta ,delta_v, v0, T, a,b):
        s_star = s0 + v * T + v * delta_v / (2 * np.sqrt(a * b))
        return a * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    
    def idm_ode( y,t, p):
        T_factors= p[0]
        a_factors = p[1]
        
        x_1 = y[0]
        v_1 = y[1]
        x_2 = y[2]
        v_2 = y[3]
        # x_3 = y[4]
        # v_3 = y[5]
        # x_4 = y[6]
        # v_4 = y[7]
        # x_5 = y[8]
        # v_5 = y[9]
        s_1 = 1000
        delta_v_1 = 0
        v0_1 = v0_lead
        s_2 = x_1 - x_2
        delta_v_2 = v_2 - v_1
        v0_2 = v0_follow
        # s_3 = x_2 - x_3
        # delta_v_3 = v_3 - v_2
        # v0_3 = v0_follow
        # s_4 = x_3 - x_4
        # delta_v_4 = v_4 - v_3
        # v0_4 = v0_follow
        # s_5 = x_4 - x_5
        # delta_v_5 = v_5 - v_4
        # v0_5 = v0_follow

        T_1 = T_factors
        a_1 = a_factors
        T_2 = T_factors
        a_2 = a_factors
        T_3 = T_factors
        a_3 = a_factors
        T_4 = T_factors
        a_4 = a_factors
        T_5 = T_factors
        a_5 = a_factors

        a_1 = acceleration(v_1, s_1,s0,delta ,delta_v_1, v0_1, T_1, a_1,b)
        a_2 = acceleration(v_2, s_2,s0,delta ,delta_v_2, v0_2, T_2, a_2,b)
        # a_3 = acceleration(v_3, s_3,s0,delta ,delta_v_3, v0_3, T_3, a_3,b)
        # a_4 = acceleration(v_4, s_4,s0,delta ,delta_v_4, v0_4, T_4, a_4,b)
        # a_5 = acceleration(v_5, s_5,s0,delta ,delta_v_5, v0_5, T_5, a_5,b)

        # dydt = [v_1, a_1, v_2, a_2, v_3, a_3, v_4, a_4, v_5, a_5]
        dydt = [v_1, a_1, v_2, a_2]
        return dydt

    for i in range(n_cars):
        data[i,:] = observed_data[observed_data['car_id']==i]['position'].values
        data[i+1,:] =  observed_data[observed_data['car_id']==i]['velocity'].values
    
    ode_model = pm.ode.DifferentialEquation(func = idm_ode, times = t_eval, n_states=4, n_theta=2, t0=0)
  
    with pm.Model() as model:
        # Specify prior distributions for some of our model parameters
        sigma = pm.HalfCauchy("sigma", 1)
        x0 = pm.Normal("x0", x0, 1)
        # If we know one of the parameter values, we can simply pass the value.
        ode_solution = ode_model(y0=x0, theta=[T_factors, a_factors])
        # The ode_solution has a shape of (n_times, n_states)
        Y = pm.Normal("Y", mu=ode_solution, sigma=sigma, observed=data[0:4,:].T)
        prior = pm.sample_prior_predictive()
        trace = pm.sample(2000, tune=1000, cores=5)
        posterior_predictive = pm.sample_posterior_predictive(trace)
        az.plot_trace(trace)
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    posterior.to_dataframe().to_csv("test.csv")
# def simulator_forward_model(rng, sigma,size=None):
#     sol = odeint(idm_ode, t_span, x0, t_eval=t_eval, vectorized=True, args=(args))
#     positions = sol.y[::2, :].flatten()
#     velocities = sol.y[1::2, :].flatten()
#     mu = np.vstack((positions, velocities)).T
#     return rng.normal(mu, sigma)

# with pm.Model() as model:
#     # Priors
#     sigma = pm.HalfNormal("sigma", 10)
#     # ode_solution
#     pm.Simulator(
#         "Y_obs",
#         simulator_forward_model,
#         params=(sigma),
#         epsilon=1,
#         observed=data[["position", "velocity"]].values,
#     )

# sampler = "SMC_epsilon=1"
# draws = 2000
# with model:
#     trace_SMC_e1 = pm.sample_smc(draws=draws, progressbar=False)
# trace = trace_SMC_e1
# fig, ax = plt.subplots(figsize=(12, 4))
# num_samples = 10
# trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
# print(trace_df)