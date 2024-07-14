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
plot_idm_results(results_df)
    # fig, axes = plt.subplots(3, 1, sharex=True)
    # axes = plot_idm_results(axes,t_eval,results_positions,results_velocities,accelerations ,n_simulations, n_cars,stochastic)
    # plt.show()
# # import plotnine
# # import ggplot
# import pandas as pd

# # Initial conditions for four vehicles: positions and velocities
# n_cars = 4
# x0 = np.zeros(n_cars * 2)
# x0[0] = -20  # initial position of the first vehicle
# x0[2] = -40  # initial position of the second vehicle
# x0[4] = -60  # initial position of the third vehicle
# x0[6] = -80  # initial position of the fourth vehicle
# x0[1] = 30  # initial velocity of the first vehicle (m/s)
# x0[3] = 0  # initial velocity of the second vehicle (m/s)
# x0[5] = 0  # initial velocity of the third vehicle (m/s)
# x0[7] = 0  # initial velocity of the fourth vehicle (m/s)

# # IDM acceleration function with different target speeds

# # Time span for the simulation
# t_max = 1000  # total simulation time (s)
# t_span = (0, t_max)
# t_eval = np.linspace(0, t_max, 1000)

# # Solving the ODE for four vehicles over 1000 seconds with new parameters
# sol = scipy.integrate.solve_ivp(idm.idm_ode, t_span, x0, t_eval=t_eval, vectorized=True)

# # Extracting positions and velocities for four vehicles
# positions = sol.y[::2, :]
# velocities = sol.y[1::2, :]

# # Reduce the data to the first 150 steps
# steps = 150

# # Extracting the reduced positions and velocities
# reduced_positions = positions[:, :steps]
# reduced_velocities = velocities[:, :steps]
# reduced_time = sol.t[:steps]

# # Finding the time when the distance between the vehicles is less than 5 meters within the reduced steps
# reduced_distance_1_2 = reduced_positions[0, :] - reduced_positions[1, :]
# reduced_distance_2_3 = reduced_positions[1, :] - reduced_positions[2, :]
# reduced_distance_3_4 = reduced_positions[2, :] - reduced_positions[3, :]
# reduced_time_less_than_5_1_2 = reduced_time[np.where(reduced_distance_1_2 < 5)[0][0]]
# reduced_time_less_than_5_2_3 = reduced_time[np.where(reduced_distance_2_3 < 5)[0][0]]
# reduced_time_less_than_5_3_4 = reduced_time[np.where(reduced_distance_3_4 < 5)[0][0]]

# # Plotting the reduced positions and velocities of each vehicle over time with vertical lines indicating when the distance was less than 5 meters
# fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# # Position subplot
# for i in range(n_cars):
#     axs[0].plot(reduced_time, reduced_positions[i, :], label=f'Car {i+1}')
# axs[0].axvline(x=reduced_time_less_than_5_1_2, color='r', linestyle='--')
# axs[0].axvline(x=reduced_time_less_than_5_2_3, color='g', linestyle='--')
# axs[0].axvline(x=reduced_time_less_than_5_3_4, color='b', linestyle='--')
# axs[0].set_xlabel('Time (s)')
# axs[0].set_ylabel('Position (m)')
# axs[0].set_title('IDM Simulation: Positions of Cars Over Time (First 150 steps)')
# axs[0].legend()
# axs[0].grid()

# # Velocity subplot
# for i in range(n_cars):
#     axs[1].plot(reduced_time, reduced_velocities[i, :], label=f'Car {i+1}')
# axs[1].axvline(x=reduced_time_less_than_5_1_2, color='r', linestyle='--')
# axs[1].axvline(x=reduced_time_less_than_5_2_3, color='g', linestyle='--')
# axs[1].axvline(x=reduced_time_less_than_5_3_4, color='b', linestyle='--')
# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('Velocity (m/s)')
# axs[1].set_title('IDM Simulation: Velocities of Cars Over Time (First 150 steps)')
# axs[1].legend()
# axs[1].grid()

# plt.tight_layout()
# plt.show()


# # Calculate the distances to the first vehicle for each simulation, excluding the first vehicle
# distances_to_first_vehicle = []
# for i in range(n_simulations):
#     distances = results_positions[i] - results_positions[i][0, :]
#     distances_to_first_vehicle.append(distances[1:, :])  # Exclude the first vehicle

# # Reduce the data to the first 40 seconds
# steps = 40

# # Extracting the reduced positions and velocities, excluding the first vehicle
# reduced_distances = [distances[:, :steps] for distances in distances_to_first_vehicle]
# reduced_velocities = [velocities[1:, :steps] for velocities in results_velocities]
# reduced_time = t_eval[:steps]




