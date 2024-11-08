from system_model import OscillatorModel
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import json

num_samples = 200
seconds = 10
t = np.arange(0, seconds, 0.01)
data = np.zeros((num_samples, len(t), 5))

for i in range(1, num_samples):

    x = np.random.normal(0, 0.1) * np.ones((len(t),1))
    # x = np.ones((len(t),1))
    x = x.reshape(len(t),)
    w = np.random.normal(1.5, 0.1) * np.ones((len(t),1))
    # w = np.ones((len(t),1))*1.5
    w = w.reshape(len(t),)
    v = np.random.normal(0.5, 0.1) * np.ones((len(t),1))
    v = v.reshape(len(t),)

    t = t.reshape(len(t),)

    data[i, :, 0] = t
    data[i, :, 1] = x
    data[i, :, 2] = w
    data[i, :, 3] = v
    data[i, :, 4] =  np.multiply(x,np.cos(np.multiply(w,t))) + np.multiply(np.divide(v, w) ,
                                                                           np.sin(np.multiply(w,t)))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(f"Using device: {device}")

model = OscillatorModel(input_dim=5, output_dim=1,device=device,learning_rate=0.0001,n_steps=10000)

x = torch.from_numpy(data[:, :, 0:4]).float().to(device)
y = torch.from_numpy(data[:, :, -1]).float().to(device)

model.train(X=x, y=y)