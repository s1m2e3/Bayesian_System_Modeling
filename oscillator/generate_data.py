import numpy as np
import os 

num_samples = 10
seconds = 10
np.random.seed(0)

t = np.arange(0, seconds, 0.01)
data = np.zeros((num_samples, len(t), 5))

for i in range(num_samples):

    x = np.random.normal(0, 0.01) * np.ones((len(t),1))
    x = x.reshape(len(t),)
    w = np.random.normal(1.5, 0.1) * np.ones((len(t),1))
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
    
data_clean = data

data_noisy = np.random.normal(data, 0.01, data.shape)

if not os.path.exists('inputs'):
    os.makedirs('inputs')

np.save('./inputs/data_clean.npy',data_clean)
np.save('./inputs/data_noisy.npy',data_noisy)
