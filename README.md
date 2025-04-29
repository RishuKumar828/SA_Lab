,,,import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, t
from scipy.optimize import curve_fit

# Example dataset
Xi = np.array([1, 2, 3, 4, 5])  # Independent variable
Yi = np.array([2.1, 4.0, 6.1, 8.2, 10.1])  # Dependent variable
errors_Y = np.array([0.2, 0.3, 0.2, 0.4, 0.3])  # Errors in Yi (variable)

# Linear model function
def linear_model(x, m, c):
    return m * x + c

# Perform linear regression using curve_fit, accounting for errors
params, cov_matrix = curve_fit(linear_model, Xi, Yi, sigma=errors_Y, absolute_sigma=True)

slope, intercept = params
slope_error, intercept_error = np.sqrt(np.diag(cov_matrix))

# Calculate correlation coefficient
correlation_coefficient = np.corrcoef(Xi, Yi)[0, 1]

# 90% confidence interval
alpha = 0.10  # For 90% confidence
n = len(Xi)  # Number of data points
t_value = t.ppf(1 - alpha / 2, df=n - 2)  # t critical value

# Confidence interval for regression line
y_fit = linear_model(Xi, slope, intercept)
conf_interval = t_value * np.sqrt((1 / n) + ((Xi - np.mean(Xi)) ** 2) / np.sum((Xi - np.mean(Xi)) ** 2))

y_fit_upper = y_fit + conf_interval
y_fit_lower = y_fit - conf_interval

# Plot scatter plot with error bars
plt.figure(figsize=(8, 6))
plt.errorbar(Xi, Yi, yerr=errors_Y, fmt='o', label='Data points', capsize=5, color='blue')
plt.plot(Xi, y_fit, label='Regression Line', color='red')
plt.fill_between(Xi, y_fit_lower, y_fit_upper, color='red', alpha=0.2, label='90% Confidence Interval')

# Add labels, legend, and grid
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Confidence Interval')
plt.legend()
plt.grid()

# Display the plot
plt.show()

# Print results
print(f"Regression Line: Y = {slope:.4f} * X + {intercept:.4f}")
print(f"Uncertainty in Slope:    {slope_error:.4f}")
print(f"Uncertainty in Intercept:    {intercept_error:.4f}")
print(f"Correlation Coefficient: {correlation_coefficient:.4f}")
print(f"90% Confidence Interval for Slope: [{slope - t_value * slope_error:.4f}, {slope + t_value * slope_error:.4f}]")
print(f"90% Confidence Interval for Intercept: [{intercept - t_value * intercept_error:.4f}, {intercept + t_value * intercept_error:.4f}]"),,,


,,,import numpy as np
import matplotlib.pyplot as plt

# Given data
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([5, 1, 2, 4, 8, 16, 32, 64, 13, 25])
y = np.array([35, 7, 13, 24, 45, 86, 167, 328, 649, 890])

# Adding a column of ones for the intercept term (theta0)
X = np.column_stack((np.ones(x1.shape[0]), x1, x2))

# Initial theta values
theta = np.array([-283.015, 103.779, -3.913])

# Learning rate
alpha = 0.001

# Number of iterations
iterations = 1000

# Number of training examples
m = len(y)

# Cost function (mean squared error)
def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    return (1/(2*m)) * np.sum((predictions - y)**2)

cost_history = []  # To store the cost function values

for i in range(iterations):
    predictions = X.dot(theta)  # Hypothesis
    errors = predictions - y    # Errors between prediction and actual values

    # Gradient for each parameter
    gradients = (1/m) * X.T.dot(errors)

    # Update the parameters
    theta -= alpha * gradients

    # Record the cost at each iteration
    cost_history.append(compute_cost(X, y, theta))

# Final theta values after gradient descent
print(f"Final theta: {theta}")

# Plotting all graphs on a single figure
fig, axs = plt.subplots(2, 1, figsize=(6, 8))

# Cost function vs. Iterations
axs[0].plot(range(iterations), cost_history, color='blue')
axs[0].set_title('Cost Function vs. Iterations')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Cost (J(θ))')
axs[0].grid(True)

# Cost function vs. theta_1 and theta_2 (combined in one graph)
theta_1_vals = np.linspace(-10, 10, 100)
cost_vals_theta_1 = np.array([compute_cost(X, y, [theta[0], t, theta[2]]) for t in theta_1_vals])

theta_2_vals = np.linspace(-10, 10, 100)
cost_vals_theta_2 = np.array([compute_cost(X, y, [theta[0], theta[1], t]) for t in theta_2_vals])

# Plot both curves on the same graph
axs[1].plot(theta_1_vals, cost_vals_theta_1, color='red', label='Cost vs. Theta_1')
axs[1].plot(theta_2_vals, cost_vals_theta_2, color='green', label='Cost vs. Theta_2')
axs[1].set_title('Cost Function vs. Theta_1 and Theta_2')
axs[1].set_xlabel('Theta Values')
axs[1].set_ylabel('Cost (J(θ))')
axs[1].legend(loc='best')
axs[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show(),,,

import random
from scipy.stats import binomtest

def simulate_coin_tosses(n, q):
    return sum(1 for _ in range(n) if random.random() < q)

def binomial.test_coin(observed_heads, n, q_h0=0.5):
    test_result = binomtest(observed_heads, n, q_h0, alternative='two-sided')
    return test_result.pvalue

# Example usage:
n = 100  # Number of tosses
q = 0.6  # True probability of heads (for simulation)

# Simulate experiment
heads = simulate_coin_tosses(n, q)
print(f"Simulated {heads} heads out of {n} tosses")

# Perform binomial test
p_value = binomial.test_coin(heads, n)
print(f"Binomial test p-value: {p_value:.4f}")

# Interpret result
alpha = 0.05
if p_value < alpha:
    print("Reject H₀ : Evidence suggests the coin is biased")
else:
    print("Fail to reject H₀ : No evidence the coin is biased") ,,,,

,,,,,,,,,,,

import numpy as np
import random
import matplotlib.pyplot as plt

# Define number of states (M >= 2)
M = 2  # You can change this

# Define state labels (0, 1, ..., M-1)
states = list(range(M))

# Define transition probability matrix randomly
TransitionMatrix = np.random.rand(M, M)
TransitionMatrix = TransitionMatrix / TransitionMatrix.sum(axis=1, keepdims=True)

print("Transition Matrix (Step 1):\n", TransitionMatrix)

# Number of steps for Brownian motion simulation
N = 5

# Start from a random state
CurrentState = random.choice(states)

print("\nStarting state:", CurrentState)
print("Particle path:")

# Simulate Markovian Brownian motion
path = [CurrentState]
for _ in range(N-1):
    CurrentState = np.random.choice(states, p=TransitionMatrix[CurrentState])
    path.append(CurrentState)

print(path)

# Now compute Transition Matrices for multiple steps
def matrix_power(matrix, n):
    return np.linalg.matrix_power(matrix, n)

print("\nTransition Matrices for 1 to N steps:")

for steps in range(1, N+1):
    T_n = matrix_power(TransitionMatrix, steps)
    print(f"\nTransition matrix for {steps} step(s):\n", np.round(T_n, 3))

# --- PLOT the Brownian motion path ---
plt.figure(figsize=(10, 6))
plt.plot(range(N), path, marker='o', linestyle='-', color='b')
plt.title("Markovian Brownian Motion of a Particle")
plt.xlabel("Time Step")
plt.ylabel("State")
plt.grid(True)
plt.yticks(states)  # So only valid states are shown on y-axis
plt.show()
