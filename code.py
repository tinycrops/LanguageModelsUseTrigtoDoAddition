import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_stack_of_sliced_disks_with_bouncing_line(num_disks=4, num_slices=10, radius=1, gap=0.3):
    """
    Plots a stack of sliced disks with a bouncing laser beam.

    Args:
        num_disks (int): The number of disks in the stack.
        num_slices (int): The number of slices each disk is divided into.
        radius (float): The radius of each disk.
        gap (float): The gap between the disks.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for high contrast visualization
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'lime', 'pink']

    # Plot each disk in the stack
    for k in range(num_disks):
        for i in range(num_slices):
            # Define the angular range for each slice
            theta_start = i * (2 * np.pi / num_slices)
            theta_end = (i + 1) * (2 * np.pi / num_slices)
            theta = np.linspace(theta_start, theta_end, 50)
            z = np.linspace(k * gap, k * gap + 0.1, 2)  # Small thickness to make the disk look 3D
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid)
            y_grid = radius * np.sin(theta_grid)

            # Plot the surface for each slice with a different color
            ax.plot_surface(x_grid, y_grid, z_grid, color=colors[i % len(colors)], alpha=0.9)

    # Generate the bouncing line
    z_line = []
    x_line = []
    y_line = []

    # Initial position and direction
    current_z = 0
    current_x, current_y = np.random.uniform(-radius, radius), np.random.uniform(-radius, radius)

    for k in range(num_disks):
        # Ensure the laser hits each disk once
        next_z = k * gap + 0.05  # Midpoint of the current disk
        
        #Generate a random angle
        next_theta = np.random.uniform(0,2*np.pi)
        
        next_x = radius * np.cos(next_theta)
        next_y = radius * np.sin(next_theta)
        
        # Add the points to the line.  Crucially, we ADD THE *CURRENT* POINT FIRST.
        z_line.append(current_z)
        x_line.append(current_x)
        y_line.append(current_y)

        z_line.append(next_z)
        x_line.append(next_x)
        y_line.append(next_y)



        # Update current position.  This is the "bounce".
        current_z = next_z
        current_x = next_x
        current_y = next_y

    # Plot the bouncing line
    ax.plot(x_line, y_line, z_line, color='r', lw=2)

    # Set labels for the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set aspect ratio and view angle for better visualization
    ax.set_box_aspect([1, 1, num_disks])
    ax.view_init(elev=45, azim=30)

    plt.title('Stack of Sliced Disks with Bouncing Laser Beam')
    plt.show()

# Plot the stacked sliced disks with bouncing line
plot_stack_of_sliced_disks_with_bouncing_line()