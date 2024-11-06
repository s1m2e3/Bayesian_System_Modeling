from system_model import DHModel
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import json
from system_utils import read_json_file

data = read_json_file('data/data.json')

a = np.array(data['a'])
d = np.array(data['d'])
theta = np.array(data['theta'])
theta_noisy = np.array(data['theta_noisy'])
alpha = np.array(data['alpha'])
y_clean=np.array(data["positions_clean"]) 
y_offset_clean=np.array(data["positions_offset"]) 
y_noisy=np.array(data["positions_clean_noisy"]) 
y_offset_noisy=np.array(data["positions_offset_noisy"]) 
X=np.concatenate((theta_noisy, alpha), axis=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.from_numpy(a).float().to(device)
d = torch.from_numpy(d).float().to(device)
model = DHModel(theta.shape[1], y_clean.shape[1], device=device, num_joints=theta.shape[1], a=a, d=d)

print(f"Using device: {device}")

X = torch.from_numpy(X).float().to(device)
y_noisy = torch.from_numpy(y_noisy).float().to(device)
model.train(X=X, y=y_noisy)

# data.pop('a', None)
# data.pop('d', None)
# df = pd.DataFrame(data)
# print(df['theta'].shape)
# train_data = pd.read_csv('./data/Train_Data_CSV.csv')
# x=train_data[["Data_No","Flow_rate","Time","Dust_feed"]]
# x[["Flow_rate","Time","Dust_feed"]]=x[["Flow_rate","Time","Dust_feed"]]/x[["Flow_rate","Time","Dust_feed"]].max()
# y=train_data[["Data_No","Differential_pressure"]]
# y["Differential_pressure"]=y["Differential_pressure"]/y["Differential_pressure"].max()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# model = CumulativeDamageModel(input_dim=x.shape[1]-1, device=device)
# x=torch.from_numpy(x.to_numpy()).float().to(device)
# y=torch.from_numpy(y.to_numpy()).float().to(device)
# model.train(X=x, y=y)
# params = model.get_params()

# if not os.path.exists('outputs'):
#     os.makedirs('outputs')
# with open('outputs/params.json', 'w') as fp:
#     json.dump(params, fp)

