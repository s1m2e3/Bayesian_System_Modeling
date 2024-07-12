import numpy as np
from scipy.integrate import solve_ivp

def read_idm_ode_parameters():
    import yaml
    with open("car_following/idm_parameters.yaml", 'r') as file:
        data = yaml.safe_load(file)
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
    return (v0_lead, v0_follow, T_base, s0, a_base, b, delta, n_simulations, n_cars,distance_vehicle,t_min,t_max,t_step)

# Initial conditions for five vehicles: positions and velocities
def idm_initial_conditions(n_cars,distance,initial_velocity):
    x0 = np.zeros(n_cars * 2)
    for i in range(n_cars):
        x0[i*2] = -distance - distance * i  # initial position of the ith vehicle
        x0[(i*2)+1] = initial_velocity * (i == 0)  # initial velocity of the ith vehicle (m/s)
    return x0

# IDM acceleration function with stochastic T and a
def acceleration(v, s,s0,delta ,delta_v, v0, T, a,b):
    s_star = s0 + v * T + v * delta_v / (2 * np.sqrt(a * b))
    return a * (1 - (v / v0) ** delta - (s_star / s) ** 2)

# IDM ODE system
def idm_ode(t, y, n_cars,s0,delta,v0_lead, v0_follow, b, T_base, a_base, T_factors, a_factors,stochastic):
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
        T = T_base * T_factors[i] if stochastic else T_base
        a = a_base * a_factors[i] if stochastic else a_base
        a_i = acceleration(v_i, s,s0,delta ,delta_v, v0, T, a,b)
        dydt[2 * i] = v_i
        dydt[2 * i + 1] = a_i
    return dydt

# Running the simulations
def solve_idm(n_cars,n_simulations,t_span,t_eval, x0, args):
    results_positions = []
    results_velocities = []
    for _ in range(n_simulations):
        T_factors = np.random.uniform(0.8, 1.3, n_cars)
        a_factors = np.random.uniform(0.8, 1.3, n_cars)
        last_arg = args[-1]
        args = args[:-1] + (T_factors, a_factors, last_arg)
        sol = solve_ivp(idm_ode, t_span, x0, t_eval=t_eval, vectorized=True, args=(args))
        results_positions.append(sol.y[::2, :])
        results_velocities.append(sol.y[1::2, :])
    return results_positions, results_velocities


def find_accelerations(n_simulations, n_cars, velocities, t_span, t_steps):
    # Calculate the accelerations for each simulation
    accelerations = []
    for i in range(n_simulations):
        acceleration_sim = np.zeros((n_cars, t_steps))
        for j in range(n_cars):
            v_diff = np.diff(velocities[i][j, :t_steps])
            time_diff = np.diff(t_span)
            acceleration_sim[j, :] = np.concatenate(([0], v_diff / time_diff))  # First acceleration is zero
        accelerations.append(acceleration_sim)
    return accelerations

# Plotting function
# # Plotting the results with each vehicle sharing colors across the simulations, excluding the first vehicle
def plot_idm_results(axes,time,distances,velocities,accelerations ,n_simulations, n_cars,stochastic):
    
    import matplotlib.pyplot as plt
    vehicle_colors = ['b', 'g', 'r', 'c']  # Only 4 colors needed for the following vehicles
    for j in range(1, n_cars):
        for i in range(n_simulations):
            axes[0].plot(time, distances[i][j - 1, :], color=vehicle_colors[j - 1], alpha=0.7, label=f'Car {j+1}' if i == 0 else "")
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Distance to First Vehicle (m)')
    axes[0].set_title('IDM Simulation: Distance to First Vehicle Over Time (First 40 seconds)')
    axes[0].legend()
    axes[0].grid()

    # Velocity subplot
    for j in range(1, n_cars):
        for i in range(n_simulations):
            axes[1].plot(time, velocities[i][j - 1, :], color=vehicle_colors[j - 1], alpha=0.7, label=f'Car {j+1}' if i == 0 else "")
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].set_title('IDM Simulation: Velocities of Cars Over Time (First 40 seconds)')
    axes[1].legend()
    axes[1].grid()

    # Acceleration subplot
    for j in range(1, n_cars):
        for i in range(n_simulations):
            axes[2].plot(time, accelerations[i][j, :], color=vehicle_colors[j - 1], alpha=0.7, label=f'Car {j+1}' if i == 0 else "")
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].set_title('IDM Simulation: Accelerations of Cars Over Time (First 40 seconds)')
    axes[2].legend()
    axes[2].grid()
    return axes




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
