import pyro.params
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system_model  import ELMOsccilatorModel
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyro.distributions as dist
import torch
import json
import os 
import pyro


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = np.load('inputs/data_clean.npy')
x_min = torch.from_numpy(np.min(np.min(data, axis=0), axis=0)).float().to(device)[0:4]
x_max = torch.from_numpy(np.max(np.max(data, axis=0), axis=0)).float().to(device)[0:4]
data_noisy = np.load('inputs/data_noisy.npy')

x = torch.from_numpy(data[ :,:, 0:4]).float().to(device)
y = torch.from_numpy(data_noisy[:, :, -1]).float().to(device)

model = ELMOsccilatorModel(input_dim=4,hidden_dim=100, output_dim=1,device=device,learning_rate=0.0001,n_steps=2000)
x = 2*(x-x_min)/(x_max-x_min)-1

model.set_priors(X=x,y=y)

predictions = model.predict(X_new=x)
std_pred = predictions['std_dev']
mean_pred = predictions['mean']

std_data = np.std(data_noisy[:, :, -1], axis=0)
mean_data = np.mean(data_noisy[:, :, -1], axis=0)

pred_df = pd.DataFrame([mean_pred, std_pred]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/prior_prediction_elm.csv')

model.train(X=x, y=y)
predictions = model.predict(X_new=x)
std_pred = predictions['std_dev']
mean_pred = predictions['mean']

pred_df = pd.DataFrame([mean_pred, std_pred]).T
pred_df.columns = ['mean', 'std']
pred_df.to_csv('outputs/posterior_prediction_elm.csv')



# model.train(X=x, y=y)
# params = model.get_params()
# predictions = model.predict(X_new=x)

# # Use numpy.percentile to calculate HDI bounds
# std_pred = predictions['std_dev']
# mean_pred = predictions['mean']
# pred_df = pd.DataFrame([mean_pred, std_pred]).T
# pred_df.columns = ['mean', 'std']
# pred_df.to_csv('outputs/post_prediction.csv')

# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.plot(t, mean_pred, label='Post-Training Prediction', color='g')
# plt.fill_between(t, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, color='g')
# plt.legend()
# plt.show()
# if not os.path.exists('outputs'):
#     os.makedirs('outputs')
# with open('outputs/params.json', 'w') as fp:
#     json.dump(params, fp)