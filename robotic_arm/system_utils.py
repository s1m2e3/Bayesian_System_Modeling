import numpy as np
import json

def read_json_file(filename):
    """Reads a JSON file and returns the data as a dictionary."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def dh_transform_matrix_numpy(theta, d, a, alpha):
    """
    Generate a batch of DH Transformation matrices based on DH parameters.
    theta and alpha are expected to be vectors of the same length.
    """
    theta = np.expand_dims(theta, axis=0) 
    alpha = np.expand_dims(alpha, axis=0)
    theta = np.deg2rad(theta)  # Convert to radians for calculations
    alpha = np.deg2rad(alpha)  # Convert to radians for calculations

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)


    return np.stack([
        np.stack([cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta], axis=-1),
        np.stack([sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta], axis=-1),
        np.stack([zeros, sin_alpha, cos_alpha, d * ones], axis=-1),
        np.stack([zeros, zeros, zeros, ones], axis=-1)
    ], axis=-2)

def get_Tij(T_i, T_j):
    """Calculate the transformation matrix from frame i to frame j."""
    return T_i @ T_j

def get_positions(T_i, dims):
    """Calculate the position of the end-effector in frame i."""
    return T_i[:,:dims, dims]

def noise_data(data, observation_noise):
    return data + np.random.normal(0, observation_noise, data.shape)


