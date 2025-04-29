# Create a grid of theta and phi values
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm


theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Function to plot spherical harmonics
def plot_spherical_harmonics(l, m):
    # Calculate the spherical harmonics
    Y_lm = sph_harm(m, l, phi, theta)
    
    # Calculate the probability density
    probability_density = np.abs(Y_lm) ** 2
    
    # Convert to Cartesian coordinates for plotting
    x = np.sin(theta) * np.cos(phi) * probability_density
    y = np.sin(theta) * np.sin(phi) * probability_density
    z = np.cos(theta) * probability_density
    
    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(probability_density), rstride=1, cstride=1, alpha=0.7)
    
    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Spherical Harmonics: l={l}, m={m}')
    
    plt.show()

# Example: Plot for l=2, m=1
plot_spherical_harmonics(5, 4)
