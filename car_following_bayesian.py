import numpy as np
import sunode
import sunode.wrappers.as_pytensor
import pymc as pm
import matplotlib.pyplot as plt
from car_following.idm import read_idm_ode_parameters,  idm_initial_conditions, idm_ode, plot_idm_results,find_accelerations
from utils.diff_eq import simulation_params
import pandas as pd
from scipy.integrate import solve_ivp
import pytensor as pt
import arviz as az

observed_data = pd.read_csv('./car_following/car_following_results.csv')
observed_data=observed_data[(observed_data['stochastic']==True)]

observed_data[['position','velocity']] =  np.random.normal(observed_data[['position','velocity']],2)
# observed_data = observed_data[['position','velocity','car_id','t']]
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
x0 = x0[0:6]
n_simulations = 70
data = np.zeros((n_simulations,sol.y.shape[0],sol.y.shape[1]))
def acceleration(v, s,s0,delta ,delta_v, v0, T, a,b):
    s_star = s0 + v * T + v * delta_v / (2 * np.sqrt(a * b))
    return a * (1 - (v / v0) ** delta - (s_star / s) ** 2)


for j in range(n_simulations):
    for i in range(n_cars):
        data[j,2*i,:] = observed_data[(observed_data['car_id']==i) & ((observed_data['simulation_id']==j)) ]['position'].values
        data[j,2*i+1,:] =  observed_data[(observed_data['car_id']==i) & ((observed_data['simulation_id']==j)) ]['velocity'].values

def idm_ode(t, y, p):
    """Right hand side of IDM equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """

    x_1 = y.position_0
    v_1 = y.velocity_0
    x_2 = y.position_1
    v_2 = y.velocity_1
    x_3 = y.position_2
    v_3 = y.velocity_2
    
    s_1 = 1000
    delta_v_1 = 0
    v0_1 = p.lead
    s_2 = x_1 - x_2
    delta_v_2 = v_2 - v_1
    v0_2 = p.follow
    s_3 = x_2 - x_3
    delta_v_3 = v_3 - v_2
    v0_3 = p.follow

    s_star_1 = p.s0 + v_1 * p.T_factors + v_1 * delta_v_1 / (2 * np.sqrt(a_factors * b))
    a_1 = a_factors * (1 - (v_1/v0_1) ** p.delta - (s_star_1/s_1) ** 2)

    s_star_2 = p.s0 + v_2 * p.T_factors + v_2 * delta_v_2 / (2 * np.sqrt(a_factors * b))
    a_2 = a_factors * (1 - (v_2/v0_2) ** p.delta - (s_star_2/s_2) ** 2)

    s_star_3 = p.s0 + v_3 * p.T_factors + v_3 * delta_v_3 / (2 * np.sqrt(a_factors * b))
    a_3 = a_factors * (1 - (v_3/v0_3) ** p.delta - (s_star_3/s_3) ** 2)
    # a_3 = acceleration(v_3, s_3,s0,delta ,delta_v_3, v0_3, T_3, a_3,b)
    # a_4 = acceleration(v_4, s_4,s0,delta ,delta_v_4, v0_4, T_4, a_4,b)
    # a_5 = acceleration(v_5, s_5,s0,delta ,delta_v_5, v0_5, T_5, a_5,b)

    # dydt = [v_1, a_1, v_2, a_2, v_3, a_3, v_4, a_4, v_5, a_5]
    return {
        'position_0': v_1,
        'velocity_0': a_1,
        'position_1': v_2,
        'velocity_1': a_2,
        'position_2': v_3,
        'velocity_2': a_3
    }


with pm.Model() as model:

    # Specify prior distributions for some of our model parameters
    sigma = pm.HalfCauchy("sigma", 1,shape=(3,2,num_step))
    alpha = pm.HalfCauchy("alpha", 1,shape=(3,2,num_step))
    # position_0 = pm.Normal("position_0_start", x0[0], 1)
    # position_1 = pm.Normal("position_1_start", x0[2], 1)
    # velocity_0 = pm.Normal("velocity_0_start", x0[1], 1)
    # velocity_1 = pm.Normal("velocity_1_start", x0[3], 1)
    
    position_0 = pm.Data("position_0_start", x0[0])
    position_1 = pm.Data("position_1_start", x0[2])
    position_2 = pm.Data("position_2_start", x0[4])
    velocity_0 = pm.Data("velocity_0_start", x0[1])
    velocity_1 = pm.Data("velocity_1_start", x0[3])
    velocity_2 = pm.Data("velocity_2_start", x0[5])

   
    v_follow = pm.Data('v_follow', v0_follow)
    v_lead = pm.Data('v_lead', v0_lead)
    T_factors = pm.Data('T_factors_param', T_factors)
    delta = pm.Data('delta_param', delta)
    s0 = pm.Data('s0_param', s0)

    y_hat, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(
        y0={
        # The initial conditions of the ode. Each variable
        # needs to specify a PyTensor or numpy variable and a shape.
        # This dict can be nested.
            'position_0': (position_0, ()),
            'velocity_0': (velocity_0, ()),
            'position_1': (position_1, ()),
            'velocity_1': (velocity_1, ()),
            'position_2': (position_2, ()),
            'velocity_2': (velocity_2, ())

        },
        params={
        # Each parameter of the ode. sunode will only compute derivatives
        # with respect to PyTensor variables. The shape needs to be specified
        # as well. It it infered automatically for numpy variables.
        # This dict can be nested.
            'T_factors': (T_factors, ()),
            'delta': (delta, ()),
            's0': (s0, ()),
            'follow': (v_follow, ()),
            'lead': (v_lead, ()),
            'extra': np.zeros(1),
        },
        # A functions that computes the right-hand-side of the ode using
        # sympy variables.
        rhs=idm_ode,
        # The time points where we want to access the solution
        tvals=t_eval,
        t0=t_eval[0],
    )
    
    # We can access the individual variables of the solution using the
    # variable names.
    pm.Deterministic('position_0_mu', y_hat['position_0'])
    pm.Deterministic('velocity_0_mu', y_hat['velocity_0'])
    pm.Deterministic('position_1_mu', y_hat['position_1'])
    pm.Deterministic('velocity_1_mu', y_hat['velocity_1'])
    pm.Deterministic('position_2_mu', y_hat['position_2'])
    pm.Deterministic('velocity_2_mu', y_hat['velocity_2'])
    
    # pm.Normal('position_0', mu=y_hat['position_0'], sigma=sigma[0,0,:], observed=data[0,:])
    # pm.Normal('velocity_0', mu=y_hat['velocity_0'], sigma=sigma[0,1,:], observed=data[1,:])
    # pm.Normal('position_1', mu=y_hat['position_1'], sigma=sigma[1,0,:], observed=data[2,:])
    # pm.Normal('velocity_1', mu=y_hat['velocity_1'], sigma=sigma[1,1,:], observed=data[3,:])
    # pm.Normal('position_2', mu=y_hat['position_2'], sigma=sigma[2,0,:], observed=data[4,:])
    # pm.Normal('velocity_2', mu=y_hat['velocity_2'], sigma=sigma[2,1,:], observed=data[5,:])
    # Loop over each replication
    for i in range(n_simulations):
        pm.SkewNormal(f'position_0_rep_{i}', mu=y_hat['position_0'], sigma=sigma[0, 0, :],alpha=alpha[0, 0, :], observed=data[i, 0, :])
        pm.SkewNormal(f'velocity_0_rep_{i}', mu=y_hat['velocity_0'], sigma=sigma[0, 1, :],alpha=alpha[0, 1, :], observed=data[i, 1, :])
        pm.SkewNormal(f'position_1_rep_{i}', mu=y_hat['position_1'], sigma=sigma[1, 0, :],alpha=alpha[1, 0, :], observed=data[i, 2, :])
        pm.SkewNormal(f'velocity_1_rep_{i}', mu=y_hat['velocity_1'], sigma=sigma[1, 1, :],alpha=alpha[1, 1, :], observed=data[i, 3, :])
        pm.SkewNormal(f'position_2_rep_{i}', mu=y_hat['position_2'], sigma=sigma[2, 0, :],alpha=alpha[2, 0, :], observed=data[i, 4, :])
        pm.SkewNormal(f'velocity_2_rep_{i}', mu=y_hat['velocity_2'], sigma=sigma[2, 1, :],alpha=alpha[2, 1, :], observed=data[i, 5, :])

with model:
    idata = pm.sample(tune=100, draws=100, chains=6, cores=6)
   
# posterior = idata.posterior.stack(sample=("chain", "draw")
posterior = idata.posterior.stack()
print(posterior)

posterior.position_0_mu.to_dataframe().to_csv('position_0_mu.csv')
posterior.position_1_mu.to_dataframe().to_csv('position_1_mu.csv')
posterior.position_2_mu.to_dataframe().to_csv('position_2_mu.csv')
posterior.velocity_0_mu.to_dataframe().to_csv('velocity_0_mu.csv')
posterior.velocity_1_mu.to_dataframe().to_csv('velocity_1_mu_.csv')
posterior.velocity_2_mu.to_dataframe().to_csv('velocity_2_mu_.csv')
# posterior.sigma.to_dataframe().to_csv('sigma  .csv')
