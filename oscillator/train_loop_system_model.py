import pyro.params
from system_model import OscillatorModel
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import json
import os 
import pyro

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(f"Using device: {device}")

model = OscillatorModel(input_dim=5, output_dim=1,device=device,learning_rate=0.0001,n_steps=10000)

params ={   
        "x_0_loc": 0,
        "x_0_scale": 0.01,
        "w_0_loc": 1.2,
        "w_0_scale": 0.1,
        "v_0_loc": 0.3,
        "v_0_scale": 0.1,
        "sigma_loc": -1,
        "sigma_scale": 0.001
        }

ranges = {
        "x_0_loc": [-0.2, 0.2],
        "x_0_scale": [0.01, 2],
        "w_0_loc": [-0.5, 0.5],
        "w_0_scale": [0.01, 2],
        "v_0_loc": [-0.5, 0.5],
        "v_0_scale": [0.01, 2],
        "sigma_loc": [0.5, 1.5],
        "sigma_scale": [0.01, 2]
        }

model.set_priors(params)
model.set_ranges(ranges)

data = np.load('inputs/data_clean.npy')
data_noisy = np.load('inputs/data_noisy.npy')

x = torch.from_numpy(data[:, :, 0:4]).float().to(device)
y = torch.from_numpy(data_noisy[:, :, -1]).float().to(device)

predictions = model.predict(X_new=x)
std_pred = predictions['std_dev']
mean_pred = predictions['mean']

pred_df = pd.DataFrame([mean_pred, std_pred]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/prior_prediction_system.csv')
model.train(X=x, y=y)
params = model.get_params()
predictions = model.predict(X_new=x)

# Use numpy.percentile to calculate HDI bounds
std_pred = predictions['std_dev']
mean_pred = predictions['mean']
pred_df = pd.DataFrame([mean_pred, std_pred]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/post_prediction_system.csv')

if not os.path.exists('outputs'):
    os.makedirs('outputs')
with open('outputs/params.json', 'w') as fp:
    json.dump(params, fp)