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
        
        # Sample x_0, w_0, v_0 with batch plate over all data
        # x_0 = X[:,0,1]
        
        # x_0 = x_0.unsqueeze(-1)
        # w_0 = X[:,0,2]
        # w_0 = w_0.unsqueeze(-1)
        with pyro.plate("data", X.shape[0], dim=-2):
            
            x_0 = pyro.sample("x_0", dist.Normal(0,0.1)).clamp(min=-0.1,max=0.1) 
            w_0 = pyro.sample("w_0", dist.Normal(1.5,0.01)).clamp(max=1.6,min=1.4) 
            v_0 = pyro.sample("v_0", dist.Normal(0.5,0.01)).clamp(max=0.6,min=0.4) 
            
            x_0 = torch.multiply(x_0,torch.ones((X.shape[0],X.shape[1]), device=self.device))
            w_0 = torch.multiply(w_0,torch.ones((X.shape[0],X.shape[1]), device=self.device))
            v_0 = torch.multiply(v_0,torch.ones((X.shape[0],X.shape[1]), device=self.device))

            t = X[:, :, 0]  # Assuming X[:,:,0] contains time points
            
            
            position = x_0 * torch.cos(w_0 * t) + (v_0 / w_0) * torch.sin(w_0 * t)
            
            # Use numpy.percentile to calculate HDI bounds
            std_pred = np.std(position.detach().numpy(), axis=0)
            std_data = np.std(y.detach().numpy(), axis=0)
            mean_pred = np.mean(position.detach().numpy(), axis=0)
            mean_data = np.mean(y.detach().numpy(), axis=0)
            
        # Ensure sigma is positive by using a HalfCauchy distribution
        sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))
        
        # Observation model
        with pyro.plate("data_obe", X.shape[0], dim=-2):
            pyro.sample("obs", dist.Normal(position, sigma), obs=y)

    def guide(self, X, y=None):
        '''
        Guide function with variational inference for x_0, w_0, v_0, and sigma
        '''
        # Move input data to the correct device
        X = X.to(self.device)
        
        # Variational parameters for x_0, w_0, and v_0, each dependent on the batch dimension
        # x_0, w_0, v_0 should be sampled for each data point (one per data point, not per feature)
        x_0_loc = pyro.param("x_0_loc", torch.zeros(1, device=self.device))  # [batch_size, 1]
        x_0_scale = pyro.param("x_0_scale", torch.ones(1, device=self.device), constraint=dist.constraints.positive)  # [batch_size, 1]
        
        w_0_loc = pyro.param("w_0_loc", torch.zeros(1, device=self.device))  # [batch_size, 1]
        w_0_scale = pyro.param("w_0_scale", torch.ones(1, device=self.device), constraint=dist.constraints.positive)  # [batch_size, 1]
        
        v_0_loc = pyro.param("v_0_loc", torch.zeros(1, device=self.device))  # [batch_size, 1]
        v_0_scale = pyro.param("v_0_scale", torch.ones(1, device=self.device), constraint=dist.constraints.positive)  # [batch_size, 1]
        
        # Variational parameters for sigma (shared for the entire batch)
        sigma_loc = pyro.param("sigma_loc", torch.ones(1, device=self.device))
        sigma_scale = pyro.param("sigma_scale", torch.ones(1, device=self.device), constraint=dist.constraints.positive)
        
        # Sample x_0, w_0, and v_0 for each data point (i.e., across the entire feature space)
        with pyro.plate("data", X.shape[0], dim=-2):
            x_0 = pyro.sample("x_0", dist.Normal(x_0_loc, x_0_scale))  # [batch_size, 1]
            w_0 = pyro.sample("w_0", dist.Normal(w_0_loc, w_0_scale))  # [batch_size, 1]
            v_0 = pyro.sample("v_0", dist.Normal(v_0_loc, v_0_scale))  # [batch_size, 1]
        
        # Sample sigma as a log-normal distribution (shared for all data points in the batch)
        sigma = pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))


    def get_params(self):
        """Retrieves the learned variational parameters."""
        concentration = pyro.param("concentration").item()
        scale_loc = pyro.param("scale_loc").detach().cpu().numpy().tolist()
        theta_scale_loc = pyro.param("theta_scale_loc").detach().cpu().numpy().tolist()
        return {
            "concentration": concentration,
            "scale_loc": scale_loc,
            "theta_scale_loc": theta_scale_loc
        }
    
    # def predict(self, X_new):
    #     """Generates predictions for DH model."""
    #     params = self.get_params()
    #     concentration = params["concentration"]
    #     scale_loc = torch.tensor(params["scale_loc"]).to(self.device)
    #     theta_scale_loc = torch.tensor(params["theta_scale_loc"]).to(self.device)
    #     # Compute the time-dependent shape parameter alpha(t)
    #     correlation_matrix = pyro.deterministic("correlation_matrix", dist.LKJCorrCholesky(self.num_joints, concentration).sample())
    #     scale = pyro.deterministic("scale", dist.HalfCauchy(scale_loc).sample())
    #     theta_mean = X_new[:, :self.num_joints]
    #     theta = pyro.deterministic("theta", dist.Normal(theta_mean, theta_scale_loc).sample())
    #     # Predict the mean cumulative damage using the MultivariateNormal distribution
    #     predictions = pyro.deterministic("predictions", dist.MultivariateNormal(theta, correlation_matrix).sample)

    #     return predictions

    def train(self, X, y, batch_size=32):
        """Trains the model using SVI with mini-batches."""
        num_samples = X.shape[0]
        for step in range(self.n_steps):
            perm = torch.randperm(num_samples)
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
    

class ELMDHModel(BayesianModel):
    def __init__(self, input_dim_elm, output_dim_elm, input_dim_dh, output_dim_dh, hidden_dim_elm, num_joints_dh, learning_rate=0.01, n_steps=500, device=None):
        super().__init__(input_dim_elm + input_dim_dh, learning_rate, n_steps, device)
        
        # Initialize ELM and DH models
        self.elm = ELM(input_dim=input_dim_elm, output_dim=output_dim_elm,  hidden_dim=hidden_dim_elm, learning_rate=learning_rate, n_steps=n_steps, device=device)
        self.dh = DHModel(input_dim=input_dim_dh, output_dim=output_dim_dh, num_joints=num_joints_dh, learning_rate=learning_rate, n_steps=n_steps, device=device)
        
        # Define dimensions for the combined model
        self.joint_dim = hidden_dim_elm + num_joints_dh  # Dimension of the joint output
    
    def model(self, X_elm, X_dh, y=None):
        """Defines the joint model for ELM and DHModel as a multivariate Gaussian."""
        
        # Obtain predictions from the ELM model
        hidden_layer_elm = torch.tanh(X_elm @ self.elm.input_weights + self.elm.input_bias)
        output_weights_mean_elm = pyro.sample(
            "output_weights_mean_elm", dist.Normal(torch.zeros(self.elm.hidden_dim, 1).to(self.device), torch.ones(self.elm.hidden_dim, 1).to(self.device)).to_event(2)
        )
        mean_output_elm = hidden_layer_elm @ output_weights_mean_elm
        
        # Obtain predictions from the DH model
        correlation_matrix_dh = pyro.sample("correlation_matrix_dh", dist.LKJCorrCholesky(self.dh.num_joints, 1.0))
        scale_dh = pyro.sample("scale_dh", dist.HalfCauchy(torch.ones(self.dh.num_joints).to(self.device)))
        covariance_matrix_dh = torch.diag(scale_dh) @ correlation_matrix_dh @ torch.diag(scale_dh)
        
        theta_mean_dh = X_dh[:, :self.dh.num_joints]
        theta_scale_dh = pyro.sample("theta_scale_dh", dist.HalfCauchy(torch.ones(self.dh.num_joints).to(self.device)))
        theta_dh = pyro.sample("theta_dh", dist.Normal(theta_mean_dh, theta_scale_dh).to_event(1))
        alpha_dh = X_dh[:, self.dh.num_joints:2*self.dh.num_joints]
        
        t_dh = {}
        for i in range(self.dh.num_joints):
            t_dh[i] = self.dh.dh_transform_matrix_torch(theta_dh[:, i], alpha_dh[:, i]).squeeze()
        t12_dh = torch.matmul(t_dh[0], t_dh[1])
        t23_dh = torch.matmul(t12_dh, t_dh[2])
        mean_output_dh = get_positions(t23_dh)
        
        # Combine outputs for joint distribution
        joint_mean = torch.cat([mean_output_elm, mean_output_dh], dim=-1).squeeze()
        
        # Define a covariance matrix for the joint distribution
        joint_scale = pyro.sample("joint_scale", dist.HalfCauchy(torch.ones(self.joint_dim).to(self.device)))
        joint_correlation = pyro.sample("joint_correlation", dist.LKJCorrCholesky(self.joint_dim, 1.0))
        joint_covariance_matrix = torch.diag(joint_scale) @ joint_correlation @ torch.diag(joint_scale)
        
        
        # Print device of each tensor in the joint covariance matrix
        # Multivariate Normal distribution for the joint likelihood
        with pyro.plate("data", X_elm.shape[0], dim=-2):
            pyro.sample("obs", dist.MultivariateNormal(joint_mean, joint_covariance_matrix), obs=y)

    def guide(self, X_elm, X_dh, y=None):
        """Guide function for variational inference with variational parameters for ELM and DHModel."""
        
        # Variational parameters for ELM output weights
        output_weights_mean_loc_elm = pyro.param(
            "output_weights_mean_loc_elm", torch.randn(self.elm.hidden_dim, 1).to(self.device)
        )
        output_weights_mean_scale_elm = pyro.param(
            "output_weights_mean_scale_elm", torch.ones(self.elm.hidden_dim, 1).to(self.device), 
            constraint=dist.constraints.positive
        )
        
        # Sample output weights for the ELM model
        pyro.sample("output_weights_mean_elm", dist.Normal(output_weights_mean_loc_elm, output_weights_mean_scale_elm).to_event(2))
        
        # Variational parameters for DH model
        concentration_dh = pyro.param("concentration_dh", torch.tensor(1.0), constraint=dist.constraints.positive)
        correlation_matrix_dh = pyro.sample(
            "correlation_matrix_dh",
            dist.LKJCorrCholesky(self.dh.num_joints, concentration_dh)
        )
        scale_loc_dh = pyro.param("scale_loc_dh", torch.ones(self.dh.num_joints).to(self.device), constraint=dist.constraints.positive)
        scale_dh = pyro.sample("scale_dh", dist.HalfCauchy(scale_loc_dh))
        
        theta_scale_loc_dh = pyro.param("theta_scale_loc_dh", torch.ones(self.dh.num_joints).to(self.device), constraint=dist.constraints.positive)
        theta_mean_loc_dh = X_dh[:, :self.dh.num_joints]
        pyro.sample("theta_dh", dist.Normal(theta_mean_loc_dh, theta_scale_loc_dh).to_event(1))
        
        # Variational parameters for joint distribution
        joint_scale_loc = pyro.param("joint_scale_loc", torch.ones(self.joint_dim).to(self.device), constraint=dist.constraints.positive)
        joint_correlation = pyro.sample("joint_correlation", dist.LKJCorrCholesky(self.joint_dim, 1.0))
        pyro.sample("joint_scale", dist.HalfCauchy(joint_scale_loc))

        