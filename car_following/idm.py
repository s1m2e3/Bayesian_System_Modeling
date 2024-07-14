import numpy as np
from scipy.integrate import solve_ivp
from data_types.data_types import IDMSimulation
from plotnine import aes, geom_line,geom_point ,labs
from utils.plots_conf import get_color


def read_idm_ode_parameters():
    import yaml
    with open("car_following/idm_parameters.yaml", 'r') as file:
        data = yaml.safe_load(file)['idm_parameters']
    v0_lead = data['v0_lead']
    v0_follow = data['v0_follow']
    T_base = data['T_base']
    s0 = data['s0']
    a_base = data['a_base']
    b = data['b']
    delta = data['delta']
    n_simulations = data['n_simulations']
    n_cars = data['n_cars']
    distance_vehicle = data['distance_vehicle']
    t_min = data['t_min']
    t_max = data['t_max']
    t_step = data['t_step']
    return v0_lead, v0_follow, T_base, s0, a_base, b, delta, n_simulations, n_cars,distance_vehicle,t_min,t_max,t_step

# Initial conditions for five vehicles: positions and velocities
def idm_initial_conditions(n_cars,distance_vehicle,v0_lead):
    x0 = np.zeros(n_cars * 2)
    for i in range(n_cars):
        x0[i*2] = -distance_vehicle - distance_vehicle * i  # initial position of the ith vehicle
        x0[(i*2)+1] = v0_lead * (i == 0)  # initial velocity of the ith vehicle (m/s)
    return x0

# IDM acceleration function with stochastic T and a
def acceleration(v, s,s0,delta ,delta_v, v0, T, a,b):
    s_star = s0 + v * T + v * delta_v / (2 * np.sqrt(a * b))
    return a * (1 - (v / v0) ** delta - (s_star / s) ** 2)

# IDM ODE system
def idm_ode(t, y, n_cars,s0,delta,v0_lead, v0_follow, b, T_base, a_base,stochastic, T_factors, a_factors):
    dydt = np.zeros_like(y)
    for i in range(n_cars):
        x_i = y[2 * i]
        v_i = y[2 * i + 1]
        if i == 0:
            s = 1000  # lead car (no car in front)
            delta_v = 0
            v0 = v0_lead
        else:
            x_front = y[2 * (i - 1)]
            v_front = y[2 * (i - 1) + 1]
            s = x_front - x_i  # gap to the car in front
            delta_v = v_i - v_front  # speed difference
            v0 = v0_follow
        # T = T_base * T_factors[i]*np.random.normal(1,0.2) if stochastic else T_base
        # a = a_base * a_factors[i]*np.random.normal(1,0.2) if stochastic else a_base
        T = T_base * T_factors[i] if stochastic else T_base
        a = a_base * a_factors[i] if stochastic else a_base
        a_i = acceleration(v_i, s,s0,delta ,delta_v, v0, T, a,b)
        dydt[2 * i] = v_i
        dydt[2 * i + 1] = a_i
    return dydt

# Running the simulations
def solve_idm(n_simulations,t_span,t_eval, x0 ,args):
    results = []
    n_cars = args[0]
    stochastic = args[-1]
    if not stochastic:
        n_simulations = 1
    for i in range(n_simulations):
        T_factors, a_factors = np.random.uniform(0.8, 1.3, n_cars),np.random.uniform(0.8, 1.3, n_cars)
        args = args + (T_factors, a_factors)
        sol = solve_ivp(idm_ode, t_span, x0, t_eval=t_eval, vectorized=True, args=(args))
        positions = sol.y[::2, :]
        velocities = sol.y[1::2, :]
        t = sol.t
        args = args[0:-2]
        for j in range(n_cars):
            y = np.column_stack((positions[j, :], velocities[j, :]))
            results.append(IDMSimulation(simulation_id=i, car_id=j, position=positions[j,:], velocity=velocities[j,:],t=t,y=y,stochastic=stochastic))
    return results


def find_accelerations(results):
    # Calculate the accelerations for each simulation
    for car in results:
        car.acceleration = np.gradient(car.velocity, car.t)   
    return results

# Plotting function
# # Plotting the results with each vehicle sharing colors across the simulations, excluding the first vehicle
def plot_idm_results(data):
    from utils.plots_conf import configure_plot, get_color
    import matplotlib.pyplot as plt
    data['car_id'] = data['car_id'].astype('category')
    data['t'] = data['t'].astype('float')
    for stochastic in data['stochastic'].unique():
        data_stochastic = data[data['stochastic'] == stochastic]
        for simulation in data_stochastic['simulation_id'].unique():
            data_simulation = data_stochastic[data_stochastic['simulation_id'] == simulation]
            for car in data_simulation['car_id'].unique():
                data_car = data_simulation[data_simulation['car_id'] == car]
                data_car['position'] = data_car['position'] - data_simulation[data_simulation['car_id']==0]['position']
                data.loc[(data['simulation_id'] == simulation) & (data['stochastic'] == stochastic) & (data['car_id'] == car), :] = data_car
    ##Extract the deterministic plot 
    data = data[data['car_id'] != 0]
    data_deterministic = data[data['stochastic'] == False]
    plot = configure_plot()
    plot = plot + geom_line(data=data_deterministic,size=3, mapping=aes(x='t', y='position', color=('car_id')))
    plot = get_color(plot)
    plot = plot + labs(title='IDM Simulation: Distance to Leader Over Time (First 40 seconds)', x='Time (s)', y='Position (m)')
    
    data_not_deterministic = data[(data['stochastic'] == True)]
    plot = plot + geom_point(data=data_not_deterministic,alpha=0.3 ,mapping=aes(x='t', y='position', color=('car_id')))
    plot = get_color(plot)
    plot.show()
    
    # vehicle_colors = ['b', 'g', 'r', 'c']  # Only 4 colors needed for the following vehicles
    # for j in range(1, n_cars):
    #     for i in range(n_simulations):
    #         axes[0].plot(time, distances[i][j - 1, :], color=vehicle_colors[j - 1], alpha=0.7, label=f'Car {j+1}' if i == 0 else "")
    # axes[0].set_xlabel('Time (s)')
    # axes[0].set_ylabel('Distance to First Vehicle (m)')
    # axes[0].set_title('IDM Simulation: Distance to First Vehicle Over Time (First 40 seconds)')
    # axes[0].legend()
    # axes[0].grid()

    # # Velocity subplot
    # for j in range(1, n_cars):
    #     for i in range(n_simulations):
    #         axes[1].plot(time, velocities[i][j - 1, :], color=vehicle_colors[j - 1], alpha=0.7, label=f'Car {j+1}' if i == 0 else "")
    # axes[1].set_xlabel('Time (s)')
    # axes[1].set_ylabel('Velocity (m/s)')
    # axes[1].set_title('IDM Simulation: Velocities of Cars Over Time (First 40 seconds)')
    # axes[1].legend()
    # axes[1].grid()

    # # Acceleration subplot
    # for j in range(1, n_cars):
    #     for i in range(n_simulations):
    #         axes[2].plot(time, accelerations[i][j, :], color=vehicle_colors[j - 1], alpha=0.7, label=f'Car {j+1}' if i == 0 else "")
    # axes[2].set_xlabel('Time (s)')
    # axes[2].set_ylabel('Acceleration (m/s²)')
    # axes[2].set_title('IDM Simulation: Accelerations of Cars Over Time (First 40 seconds)')
    # axes[2].legend()
    # axes[2].grid()
    # return axes




# # Adjusting IDM parameters for 4 vehicles to achieve a tighter composition with different target speeds
# v0_lead = 30  # desired velocity for the leading vehicle (m/s)
# v0_follow = 60  # desired velocity for the following vehicles (m/s)
# T = 0.1  # desired time headway (s)
# s0 = 1  # minimum spacing (m)
# a = 3  # maximum acceleration (m/s^2)
# b = 5  # comfortable deceleration (m/s^2)
# delta = 4
# n_cars = 4
# def acceleration(v, s, delta_v, v0):
#     s_star = s0 + v * T + v * delta_v / (2 * np.sqrt(a * b))
#     return a * (1 - (v / v0) ** delta - (s_star / s) ** 2)

# # ODE system
# def idm_ode(t, y):
#     dydt = np.zeros_like(y)
#     for i in range(n_cars):
#         x_i = y[2 * i]
#         v_i = y[2 * i + 1]
#         if i == 0:
#             s = 1000  # lead car (no car in front)
#             delta_v = 0
#             v0 = v0_lead
#         else:
#             x_front = y[2 * (i - 1)]
#             v_front = y[2 * (i - 1) + 1]
#             s = x_front - x_i  # gap to the car in front
#             delta_v = v_i - v_front  # speed difference
#             v0 = v0_follow
#         a_i = acceleration(v_i, s, delta_v, v0)
#         dydt[2 * i] = v_i
#         dydt[2 * i + 1] = a_i
#     return dydt
