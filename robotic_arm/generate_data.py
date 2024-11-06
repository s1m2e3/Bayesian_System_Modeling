# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import json
import os
# Define the DH transformation matrix function
from system_utils import dh_transform_matrix_numpy, get_positions, noise_data

np.random.seed(0)


def generate_theta_samples(num_samples, n_joints, range_min, range_max,std_theta):
    return np.random.uniform(range_min+3*std_theta, range_max-3*std_theta, (num_samples, n_joints))


def generate_alpha_samples(num_samples, n_joints, a):
    return np.random.normal(a, 1, (num_samples, n_joints))

def generate_data(num_samples, num_joints, a, d, theta, offset, alpha):
    positions = np.zeros((num_samples, n_joints, 3))
    a = a + offset
    t = {}
    for i in range(num_joints):
        t[i] = dh_transform_matrix_numpy(theta[:, i], d[i], a[0,i], alpha[:, i]).squeeze()

    positions[:, 0, :] = get_positions(t[0], num_joints)
    t12 = np.matmul(t[0], t[1])
    positions[:, 1, :] = get_positions(t12, num_joints)
    t23 = np.matmul(t12, t[2]) 
    positions[:, 2, :] = get_positions(t23, num_joints)

    return positions


std_theta = 2
theta_min = -90
theta_max = 90
n_joints = 3
num_samples = 1000

theta = generate_theta_samples(num_samples, n_joints, theta_min, theta_max, std_theta)
theta_noisy = noise_data(theta, std_theta)
alpha = generate_alpha_samples(num_samples, n_joints, 90)
offset_alpha = np.array([[-10, -10, -10]])

a = np.array([0.2, 0.2, 0.2])
d = np.array([1, 1, 1])

observation_noise = 0.01

# Store the end effector positions for each sample
positions_clean = generate_data(num_samples, n_joints, a, d, theta_noisy, np.zeros_like(offset_alpha), alpha)
positions_clean_noisy = noise_data(positions_clean, observation_noise)
positions_offset = generate_data(num_samples, n_joints, a, d, theta_noisy, offset_alpha, alpha)
positions_offset_noisy = noise_data(positions_offset, observation_noise) 

data = {}
data["positions_clean"] = positions_clean.tolist()
data["positions_offset"] = positions_offset.tolist()
data["positions_clean_noisy"] = positions_clean_noisy.tolist()
data["positions_offset_noisy"] = positions_offset_noisy.tolist()
data["theta"] = theta.tolist()
data["theta_noisy"] = theta_noisy.tolist()
data["alpha"] = alpha.tolist()
data["a"] = a.tolist()
data["d"] = d.tolist()


if not os.path.exists('data'):
    os.makedirs('data')

with open('data/data.json', 'w') as fp:
    json.dump(data, fp)
