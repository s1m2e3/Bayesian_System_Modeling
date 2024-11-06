import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt


# Step 1: Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Create some synthetic data and move it to the GPU
torch.manual_seed(42)
X = torch.randn(100, 1, device=device)  # 100 data points, 1 feature
true_w = torch.tensor([2.5], device=device)  # True weight
y = X @ true_w + 0.1 * torch.randn(100, device=device)  # Add noise

# Step 3: Define the Bayesian model (with priors)
def model(X, y=None):
    # Prior over weights: Normal(0, 1)
    w = pyro.sample("w", dist.Normal(torch.tensor(0.0, device=device), 
                                     torch.tensor(1.0, device=device)))
    # Ensure w is 1-dimensional
    w = w.unsqueeze(-1)  # Converts w from shape [] to [1]

    # Likelihood: Normal(X @ w, 0.1)
    mean = X @ w
    sigma = 0.1
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

# Step 4: Define the Guide (Variational Distribution)
def guide(X, y=None):
    # Variational parameters for the weight
    w_loc = pyro.param("w_loc", torch.tensor(0.0, device=device))  # Mean of q(w)
    w_scale = pyro.param("w_scale", torch.tensor(1.0, device=device), 
                         constraint=dist.constraints.positive)  # Std of q(w)
    pyro.sample("w", dist.Normal(w_loc, w_scale))

# Step 5: Set up the Optimizer and SVI (Stochastic Variational Inference)
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Step 6: Train the Variational Inference Model
n_steps = 5000
for step in range(n_steps):
    loss = svi.step(X, y)  # Perform a single gradient update
    if step % 500 == 0:
        print(f"Step {step} - Loss: {loss}")

# Step 7: Retrieve the Learned Variational Parameters
w_loc = pyro.param("w_loc").item()
w_scale = pyro.param("w_scale").item()

print(f"Learned weight mean: {w_loc}, std: {w_scale}")

# Step 8: Plot the Prediction including Mean ± 1 Standard Deviation
# Create X values for prediction
X_plot = torch.linspace(X.min(), X.max(), 100, device=device).unsqueeze(1)
y_mean = X_plot @ torch.tensor([w_loc], device=device)
y_std = X_plot @ torch.tensor([w_scale], device=device)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X.cpu(), y.cpu(), label='Data', alpha=0.6)
plt.plot(X_plot.cpu(), y_mean.cpu(), label='Prediction Mean', color='blue')
plt.fill_between(
    X_plot.cpu().squeeze(),
    (y_mean - y_std).cpu().squeeze(),
    (y_mean + y_std).cpu().squeeze(),
    color='blue', alpha=0.3, label='±1 Std Dev'
)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Prediction with Variational Inference (Mean ± 1 Std Dev)')
plt.legend()
plt.show()