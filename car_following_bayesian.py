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
    observed_data = observed_data[['position','velocity']]
    
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
    print(sol.y.shape)
    print(observed_data.shape)



    
    # ode_model = pm.ode.DifferentialEquation(func=freefall, times=times, n_states=1, n_theta=2, t0=0)

    # with pm.Model() as model:
    #     # Specify prior distributions for some of our model parameters
    #     sigma = pm.HalfCauchy("sigma", 1)
    #     gamma = pm.Lognormal("gamma", 0, 1)

    #     # If we know one of the parameter values, we can simply pass the value.
    #     ode_solution = ode_model(y0=[0], theta=[gamma, 9.8])
    #     # The ode_solution has a shape of (n_times, n_states)

    #     Y = pm.Normal("Y", mu=ode_solution, sigma=sigma, observed=yobs)

    #     prior = pm.sample_prior_predictive()
    #     trace = pm.sample(2000, tune=1000, cores=5)
    #     posterior_predictive = pm.sample_posterior_predictive(trace)
    #     az.plot_trace(trace)
    # posterior = trace.posterior.stack(sample=("chain", "draw"))
    # posterior.to_dataframe().to_csv("test.csv")
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