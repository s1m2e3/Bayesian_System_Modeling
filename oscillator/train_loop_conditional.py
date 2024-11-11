import numpy as np
import os 
import matplotlib.pyplot as plt
from system_model import CELMOsccilatorModel
import torch
import pandas as pd
import json

num_samples = 10
seconds = 10
np.random.seed(0)

t = np.arange(0, seconds, 0.01)
data = np.zeros((num_samples, len(t), 7))

for i in range(num_samples):

    x = np.random.normal(0, 0.01) * np.ones((len(t),1))
    x = x.reshape(len(t),)
    w = np.random.normal(1.5, 0.1) * np.ones((len(t),1))
    w = w.reshape(len(t),)
    v = np.random.normal(0.5, 0.1) * np.ones((len(t),1))
    
    damping_coefficient = np.random.normal(0.3, 0.1) * np.ones((len(t),1))
    damping_coefficient = damping_coefficient.reshape(len(t),)
    damped_frequency = np.power(w,2) - np.power(damping_coefficient,2)
    
    v = v.reshape(len(t),)
    t = t.reshape(len(t),)

    data[i, :, 0] = t
    data[i, :, 1] = x
    data[i, :, 2] = w
    data[i, :, 3] = v
    data[i, :, 4] = damping_coefficient
    data[i, :, 5] = damped_frequency
    data[i, :, 6] =  (np.multiply(x,np.cos(np.multiply(damped_frequency,t))) + np.multiply(np.divide(v+damping_coefficient*x, damped_frequency) ,
                                                                           np.sin(np.multiply(damped_frequency,t))))*np.exp(-damping_coefficient*t)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(f"Using device: {device}")

model = CELMOsccilatorModel(input_dim=6, output_dim=1,hidden_dim=100,device=device,learning_rate=0.0001,n_steps=30000)

params ={   
        "x_0_loc": 0,
        "x_0_scale": 0.01,
        "w_0_loc": 1.2,
        "w_0_scale": 0.1,
        "v_0_loc": 0.3,
        "v_0_scale": 0.1,
        "sigma_loc": (torch.zeros(t.shape,dtype=torch.float,device=device)-5),
        "sigma_scale": (torch.zeros(t.shape,dtype=torch.float,device=device)+0.001),
        'positions_loc': torch.from_numpy(data[:,:,-1].mean(axis=0)).float().to(device),
        'covariance_loc': torch.ones(t.shape,dtype=torch.float,device=device),
        'covariance_scale': torch.zeros(t.shape,dtype=torch.float,device=device)+0.1
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

x_min = torch.from_numpy(np.min(np.min(data, axis=0), axis=0)).float().to(device)[0:6]
x_max = torch.from_numpy(np.max(np.max(data, axis=0), axis=0)).float().to(device)[0:6]

x_elm = torch.from_numpy(data[ :,:, 0:6]).float().to(device)
x = torch.from_numpy(data[ :,:, 0:6]).float().to(device)
y_noisy = np.random.normal(data[:, :, -1],0.01)
y = torch.from_numpy(y_noisy).float().to(device)

x_elm = 2*(x_elm-x_min)/(x_max-x_min)-1


if not os.path.exists('outputs'):
    os.makedirs('outputs')

model.set_priors(params,x_elm,y)
model.set_ranges(ranges)

predictions = model.predict(X_new=x)

std_pred_system = predictions['system']['std_dev']
std_pred_elm = predictions['elm']['std_dev']
std_pred_conditional = predictions['conditional']['std_dev']

mean_pred_system = predictions['system']['mean']
mean_pred_elm = predictions['elm']['mean']
mean_pred_conditional = predictions['conditional']['mean']

pred_df = pd.DataFrame([mean_pred_system, std_pred_system]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/prior_prediction_conditional_system.csv')

pred_df = pd.DataFrame([mean_pred_elm, std_pred_elm]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/prior_prediction_conditional_elm.csv')

pred_df = pd.DataFrame([mean_pred_conditional, std_pred_conditional]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/prior_prediction_conditional_conditional.csv')

params = model.train(X=x, y=y)

predictions = model.predict(X_new=x)

std_pred_system = predictions['system']['std_dev']
std_pred_elm = predictions['elm']['std_dev']
std_pred_conditional = predictions['conditional']['std_dev']

mean_pred_system = predictions['system']['mean']
mean_pred_elm = predictions['elm']['mean']
mean_pred_conditional = predictions['conditional']['mean']

pred_df = pd.DataFrame([mean_pred_system, std_pred_system]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/posterior_prediction_conditional_system.csv')

pred_df = pd.DataFrame([mean_pred_elm, std_pred_elm]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/posterior_prediction_conditional_elm.csv')

pred_df = pd.DataFrame([mean_pred_conditional, std_pred_conditional]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/posterior_prediction_conditional_conditional.csv')

with open('outputs/params_conditional.json', 'w') as fp:
    json.dump(params, fp)