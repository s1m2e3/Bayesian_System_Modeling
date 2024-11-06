import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

class BayesianModel:
    def __init__(self, input_dim, learning_rate=0.01, n_steps=5000, device=None, batch_size=32):
        # Device setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Hyperparameters
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        # Initialize optimizer and SVI engine
        self.optimizer = Adam({"lr": self.learning_rate})
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())

    def model(self, X, y=None):
        """Abstract method to define the model. Should be implemented by child classes."""
        raise NotImplementedError("Model must be defined in child class.")

    def guide(self, X, y=None):
        """Abstract method to define the guide. Should be implemented by child classes."""
        raise NotImplementedError("Guide must be defined in child class.")

    def train(self, X, y):
         """Trains the model using SVI with mini-batches."""
         num_samples = X.shape[0]  # Total number of samples

         for step in range(self.n_steps):
            # Shuffle the data at the start of each epoch (optional but recommended)
            perm = torch.randperm(num_samples)  # Random permutation of indices
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Loop through mini-batches
            for i in range(0, num_samples, self.batch_size):
                # Select the mini-batch
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                # Perform a single SVI step on the mini-batch
                loss = self.svi.step(X_batch, y_batch)
                print(f"Step {step} - Loss: {loss}")
            torch.cuda.empty_cache()
            # Print the loss every 500 steps
            if step % 500 == 0:
                print(f"Step {step} - Loss: {loss}")

    def train_dictionary(self, X, y):
         """Trains the model using SVI with mini-batches."""
         num_samples = X.shape[0]  # Total number of samples

         for step in range(self.n_steps):
            # Shuffle the data at the start of each epoch (optional but recommended)
            perm = torch.randperm(num_samples)  # Random permutation of indices
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Loop through mini-batches
            for i in range(0, num_samples, self.batch_size):
                # Select the mini-batch
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                # Perform a single SVI step on the mini-batch
                loss = self.svi.step(X_batch, y_batch)
                print(f"Step {step} - Loss: {loss}")
            torch.cuda.empty_cache()
            # Print the loss every 500 steps
            if step % 500 == 0:
                print(f"Step {step} - Loss: {loss}")


    def get_params(self):
        """Abstract method to retrieve parameters. Should be overridden if necessary."""
        raise NotImplementedError("get_params should be defined in child class.")

    def predict(self, X_new):
        """Abstract method to generate predictions. Should be implemented by child classes."""
        raise NotImplementedError("Predict must be defined in child class.")

    def plot_predictions(self, X, y, X_new):
        """Plots the data, prediction mean, and uncertainty."""
        y_mean, y_std = self.predict(X_new)

        plt.figure(figsize=(10, 6))
        plt.scatter(X.cpu(), y.cpu(), label='Data', alpha=0.6)
        plt.plot(X_new.cpu(), y_mean.cpu(), label='Prediction Mean', color='blue')
        plt.fill_between(
            X_new.cpu().squeeze(),
            (y_mean - y_std).cpu().squeeze(),
            (y_mean + y_std).cpu().squeeze(),
            color='blue', alpha=0.3, label='±1 Std Dev'
        )
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Prediction with Variational Inference (Mean ± 1 Std Dev)')
        plt.legend()
        plt.show()


class ELM(BayesianModel):
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=0.01, n_steps=5000, device=None, batch_size=32):
        super().__init__(input_dim, learning_rate, n_steps, device, batch_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Randomly initialize the input weights and biases
        self.input_weights = torch.randn(input_dim, hidden_dim).to(self.device)
        self.input_bias = torch.randn(hidden_dim).to(self.device)

    def model(self, X, y=None):
        """Defines the probabilistic model for ELM with Bayesian output weights."""

        # Perform hidden layer transformation
        hidden_layer = torch.tanh(X @ self.input_weights + self.input_bias)

        # Define a Normal prior for the output weights with zero mean and unit variance
        output_weights_mean_mean = torch.zeros(self.hidden_dim, self.output_dim).to(self.device)
        output_weights_scale_mean = torch.ones(self.hidden_dim, self.output_dim).to(self.device)
        output_weights_mean = pyro.sample("output_weights_mean", dist.Normal(output_weights_mean_mean, output_weights_scale_mean).to_event(2))

        # Compute the output as a probabilistic mean and variance
        mean_output = hidden_layer @ output_weights_mean

        # Define a Normal prior for the output weights with zero mean and unit variance
        output_weights_mean_var = torch.zeros(self.hidden_dim, self.output_dim).to(self.device)
        output_weights_scale_var = torch.ones(self.hidden_dim, self.output_dim).to(self.device)
        output_weights_var = pyro.sample("output_weights_var", dist.Normal(output_weights_mean_var, output_weights_scale_var).to_event(2))

        var_output = hidden_layer @ output_weights_var
        # Likelihood function for the observed data
        with pyro.plate("data", X.shape[0]):
            pyro.sample("obs", dist.Normal(mean_output.squeeze(), var_output.sqrt()).to_event(1), obs=y)

    def guide(self, X, y=None):
        """Guide function for variational inference with separate variational parameters for output weights."""

        # Variational parameters for the mean of output weights
        output_weights_mean_loc = pyro.param(
            "output_weights_mean_loc", torch.randn(self.hidden_dim, self.output_dim).to(self.device)
        )
        output_weights_mean_scale = pyro.param(
            "output_weights_mean_scale", torch.ones(self.hidden_dim, self.output_dim).to(self.device), 
            constraint=dist.constraints.positive
        )
        
        # Sample output weights for the mean from variational distribution
        output_weights_mean = pyro.sample(
            "output_weights_mean", 
            dist.Normal(output_weights_mean_loc, output_weights_mean_scale).to_event(2)
        )

        # Variational parameters for the variance of output weights
        output_weights_var_loc = pyro.param(
            "output_weights_var_loc", torch.randn(self.hidden_dim, self.output_dim).to(self.device)
        )
        output_weights_var_scale = pyro.param(
            "output_weights_var_scale", torch.ones(self.hidden_dim, self.output_dim).to(self.device), 
            constraint=dist.constraints.positive
        )
        
        # Sample output weights for the variance from variational distribution
        output_weights_var = pyro.sample(
            "output_weights_var", 
            dist.Normal(output_weights_var_loc, output_weights_var_scale).to_event(2)
        )


    def get_params(self):
        """Retrieves the learned variational parameters."""
        output_weights_mean_loc = pyro.param("output_weights_mean_loc").detach().cpu().numpy()
        output_weights_mean_scale = pyro.param("output_weights_mean_scale").detach().cpu().numpy()
        output_weights_var_loc = pyro.param("output_weights_var_loc").detach().cpu().numpy()
        output_weights_var_scale = pyro.param("output_weights_var_scale").detach().cpu().numpy()
        
        return {
            "output_weights_mean_loc": output_weights_mean_loc,
            "output_weights_mean_scale": output_weights_mean_scale,
            "output_weights_var_loc": output_weights_var_loc,
            "output_weights_var_scale": output_weights_var_scale
        }
    
    def predict(self, X_new):
        """Generates predictions for ELM with uncertainty (mean and variance)."""
        # Get the learned parameters
        params = self.get_params()
        output_weights_mean_loc = torch.tensor(params["output_weights_mean_loc"]).to(self.device)
        output_weights_scale_loc = torch.tensor(params["output_weights_scale_loc"]).to(self.device)
        output_weights_var_loc = torch.tensor(params["output_weights_var_loc"]).to(self.device)
        output_weights_var_scale = torch.tensor(params["output_weights_var_scale"]).to(self.device)

        # Perform hidden layer transformation
        hidden_layer = torch.tanh(X_new @ self.input_weights + self.input_bias)

        # Sample the output weights and calculate output
        output_weights_mean = dist.Normal(output_weights_mean_loc, output_weights_scale_loc).sample()
        output_weights_var = dist.Normal(output_weights_var_loc, output_weights_var_scale).sample()
        
        mean_output = hidden_layer @ output_weights_mean
        var_output = hidden_layer @ output_weights_var

        output = dist.Normal(mean_output.squeeze(), var_output.sqrt().squeeze()).sample()

        return output
