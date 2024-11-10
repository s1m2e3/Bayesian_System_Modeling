import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from variational_inference.bayesian_model import BayesianModel, ELM
import torch
import pyro
import pyro.distributions as dist
from torch import cos, sin
import matplotlib.pyplot as plt
import numpy as np
from torchmin import minimize

class OscillatorModel(BayesianModel):
    def __init__(self, input_dim, output_dim, learning_rate=0.01, n_steps=500, device=None):
        super().__init__(input_dim, learning_rate, n_steps, device)
                
        self.device = device
     
     
    def model(self, X, y=None):
        '''
        Model the function: x(t) = x_0 cos(w_0*t) + v_0/w_0 sin(w_0*t)
        '''
        # Move input data to the correct device
        X = X.to(self.device)
        
        with pyro.plate("data", X.shape[0], dim=-2):
            
            x_0 = pyro.sample("x_0", dist.Normal(0,0.05)).to(self.device)
            w_0 = pyro.sample("w_0", dist.Normal(1.4,0.1)).to(self.device)
            v_0 = pyro.sample("v_0", dist.Normal(0.5,0.2)).to(self.device)
                 
            x_0 = torch.multiply(x_0,torch.ones((X.shape[0],X.shape[1]), device=self.device))
            w_0 = torch.multiply(w_0,torch.ones((X.shape[0],X.shape[1]), device=self.device))
            v_0 = torch.multiply(v_0,torch.ones((X.shape[0],X.shape[1]), device=self.device))
            t = X[:, :, 0].to(self.device)  # Assuming X[:,:,0] contains time points
            
            position = x_0 * torch.cos(w_0 * t) + (v_0 / w_0) * torch.sin(w_0 * t)
            
        # Ensure sigma is positive by using a HalfCauchy distribution
        sigma = pyro.sample("sigma", dist.HalfCauchy(0.5,1)).to(self.device).clamp(min=0.01)
        # Observation model
        with pyro.plate("data_obe", X.shape[0], dim=-2):
            pyro.sample("obs", dist.Normal(position, sigma), obs=y).to(self.device)

    def guide(self, X, y=None):
        
        '''
        Guide function with variational inference for x_0, w_0, v_0, and sigma
        '''

        # Move input data to the correct device
        X = X.to(self.device)

        x_0_loc = pyro.param("x_0_loc").to(self.device)
        x_0_scale = pyro.param("x_0_scale").to(self.device)
        w_0_loc = pyro.param("w_0_loc").to(self.device)
        w_0_scale = pyro.param("w_0_scale").to(self.device)
        v_0_loc = pyro.param("v_0_loc").to(self.device)
        v_0_scale = pyro.param("v_0_scale").to(self.device)
        sigma_loc = pyro.param("sigma_loc").to(self.device)
        sigma_scale = pyro.param("sigma_scale").to(self.device)

        # Sample x_0, w_0, and v_0 for each data point (i.e., across the entire feature space)
        with pyro.plate("data", X.shape[0], dim=-2):
            x_0 = pyro.sample("x_0", dist.Normal(x_0_loc, x_0_scale)).to(self.device)  # [batch_size, 1]
            w_0 = pyro.sample("w_0", dist.Normal(w_0_loc, w_0_scale)).to(self.device)  # [batch_size, 1]
            v_0 = pyro.sample("v_0", dist.Normal(v_0_loc, v_0_scale)).to(self.device)  # [batch_size, 1]
        
        # Sample sigma as a log-normal distribution (shared for all data points in the batch)
        sigma = pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale)).to(self.device)

    def get_params(self):
        """Retrieves the learned variational parameters."""
        x_0_loc = pyro.param("x_0_loc").detach().cpu().numpy().tolist()
        x_0_scale = pyro.param("x_0_scale").detach().cpu().numpy().tolist()
        w_0_loc = pyro.param("w_0_loc").detach().cpu().numpy().tolist()
        w_0_scale = pyro.param("w_0_scale").detach().cpu().numpy().tolist()
        v_0_loc = pyro.param("v_0_loc").detach().cpu().numpy().tolist()
        v_0_scale = pyro.param("v_0_scale").detach().cpu().numpy().tolist()
        sigma_loc = pyro.param("sigma_loc").detach().cpu().numpy().tolist()
        sigma_scale = pyro.param("sigma_scale").detach().cpu().numpy().tolist()
        return {
            "x_0_loc": x_0_loc,
            "x_0_scale": x_0_scale,
            "w_0_loc": w_0_loc,
            "w_0_scale": w_0_scale,
            "v_0_loc": v_0_loc,
            "v_0_scale": v_0_scale,
            "sigma_loc": sigma_loc,
            "sigma_scale": sigma_scale
        }
    
    def predict(self, X_new):
        """ Generates predictions for the oscillator model using learned parameters."""
        # Move new data to the correct device
        X_new = X_new.to(self.device)
        
        # Retrieve learned parameters
        params = self.priors
        x_0_loc = pyro.param("x_0_loc").detach().cpu()
        x_0_scale = pyro.param("x_0_scale").detach().cpu()
        w_0_loc = pyro.param("w_0_loc").detach().cpu()
        w_0_scale = pyro.param("w_0_scale").detach().cpu()
        v_0_loc = pyro.param("v_0_loc").detach().cpu()
        v_0_scale = pyro.param("v_0_scale").detach().cpu()
        sigma_loc = pyro.param("sigma_loc").detach().cpu()
        
        # sigma_loc = torch.abs(sigma_loc)
        sigma_scale = pyro.param("sigma_scale").detach().cpu()
        # Sample x_0, w_0, and v_0 from their learned normal distributions
        
        x_0 = torch.normal(x_0_loc, x_0_scale,size=(X_new.shape[0],1))*torch.ones(X_new.shape[0],X_new.shape[1])
        w_0 = torch.normal(w_0_loc, w_0_scale,size=(X_new.shape[0],1))*torch.ones(X_new.shape[0],X_new.shape[1])
        v_0 = torch.normal(v_0_loc, v_0_scale,size=(X_new.shape[0],1))*torch.ones(X_new.shape[0],X_new.shape[1])
        
        sigma = torch.distributions.LogNormal(sigma_loc, sigma_scale).sample((X_new.shape[0],X_new.shape[1]))
        
        # Time points from X_new
        t = X_new[:, :, 0]  # Assuming X_new[:, :, 0] contains time points
        t = t.cpu()
        # Calculate position based on sampled x_0, w_0, and v_0
        position = x_0 * torch.cos(w_0 * t) + (v_0 / w_0) * torch.sin(w_0 * t)
        
        position = torch.normal(position, 0.01)
        # Calculate summary statistics
        mean_pred = position.mean(dim=0).detach().cpu().numpy()
        std_pred = position.std(dim=0).detach().cpu().numpy()
        # Package predictions in a dictionary for easy access
        predictions = {
            "mean": mean_pred,
            "std_dev": std_pred,
            "position_samples": position.detach().cpu().numpy()
        }

        return predictions

    def train(self, X, y, batch_size=32):
        """Trains the model using SVI with mini-batches."""
        num_samples = X.shape[0]
        for step in range(self.n_steps):
            perm = torch.randperm(num_samples).to(self.device)
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i + batch_size]
                X_batch = X[idx,:]
                y_batch = y[idx,:]
                
                # Perform a single SVI step on the mini-batch
                loss = self.svi.step(X_batch, y_batch)
            
            # print(f"Step {step} - Loss: {loss}")
   
            torch.cuda.empty_cache()
            # Print the loss every 500 steps
            if step % 50 == 0:
                print(f"Step {step} - Loss: {loss}")
    
    def set_priors(self,params):

        self.priors = params
        pyro.param('x_0_loc',torch.tensor(params['x_0_loc'],dtype=torch.float, device=self.device))  
        pyro.param('x_0_scale', torch.tensor(params['x_0_scale'],dtype=torch.float, device=self.device), constraint=dist.constraints.positive)  
        pyro.param('w_0_loc', torch.tensor(params['w_0_loc'],dtype=torch.float, device=self.device))  
        pyro.param('w_0_scale', torch.tensor(params['w_0_scale'],dtype=torch.float, device=self.device), constraint=dist.constraints.positive)  
        pyro.param('v_0_loc', torch.tensor(params['v_0_loc'],dtype=torch.float, device=self.device))  
        pyro.param('v_0_scale', torch.tensor(params['v_0_scale'],dtype=torch.float, device=self.device), constraint=dist.constraints.positive) 
        pyro.param('sigma_loc', torch.tensor(params['sigma_loc'],dtype=torch.float, device=self.device) )
        pyro.param('sigma_scale', torch.tensor(params['sigma_scale'],dtype=torch.float, device=self.device), constraint=dist.constraints.positive)

    def set_ranges(self,ranges):
        self.ranges = ranges

class ELMOsccilatorModel(ELM):
        def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=0.01, n_steps=5000, device=None, batch_size=32):
                super().__init__(input_dim, output_dim, hidden_dim, learning_rate, n_steps, device, batch_size)
                
              
        def model(self, X, y=None):
                
                
                mean = pyro.param('mean').to(self.device)
                scale = pyro.param("scale", torch.tensor(1.0, device=self.device), constraint=dist.constraints.greater_than(1e-3))
                scale = torch.abs(scale)

                # Likelihood function for the observed data
                with pyro.plate("data", X.shape[0]):
                        pyro.sample("obs", dist.Normal(mean,scale).to_event(1), obs=y)

        
                

        def train(self, X, y, batch_size=32):
                """Trains the model using SVI with mini-batches."""
                num_samples = X.shape[0]
                for step in range(self.n_steps):
                        perm = torch.randperm(num_samples).to(self.device)
                        for i in range(0, num_samples, batch_size):
                                idx = perm[i:i + batch_size]
                                X_batch = X[idx,:]
                                y_batch = y[idx,:]
                                # Perform a single SVI step on the mini-batch
                                loss = self.svi.step(X_batch, y_batch)
                 
                        torch.cuda.empty_cache()
                        # Print the loss every 500 steps
                        if step % 50 == 0:
                                print(f"Step {step} - Loss: {loss}") 

        def set_priors(self,X,y):

                # Perform hidden layer transformation
                hidden_layer = torch.tanh(torch.matmul(X,self.input_weights)+ self.input_bias)
                mean_hidden_layer = hidden_layer.mean(dim=0)
                y_mean = y.mean(dim=0)
                betas_mean = torch.matmul(torch.linalg.pinv(mean_hidden_layer) , y_mean)
                residuals = torch.square(y - mean_hidden_layer@betas_mean)
                
                def sigma_objective(sigma):
                        return torch.sum(residuals/(2*torch.square(sigma)) + torch.log(2*torch.pi*torch.square(sigma))/2)

                sigma_0 = torch.ones(residuals.shape[1],device=self.device)*0.1
                sigma = minimize(sigma_objective, sigma_0,method='bfgs')
                sigma = sigma.x
                betas_sigma = torch.matmul(torch.linalg.pinv(mean_hidden_layer) , sigma)
                
                mean_pred = mean_hidden_layer@betas_mean
                mean_sigma = mean_hidden_layer@betas_sigma

                pyro.param("mean", mean_pred.to(self.device))
                pyro.param("scale", mean_sigma.to(self.device))
                         

        def predict(self, X_new):        

                mean = pyro.param('mean').to(self.device)
                sigma = pyro.param('scale').to(self.device)
                # Calculate summary statistics
                mean_pred = mean.detach().cpu().numpy()
                std_pred = sigma.detach().cpu().numpy()
                # Package predictions in a dictionary for easy access
                predictions = {
                "mean": mean_pred,
                "std_dev": std_pred
                }

                return predictions


