#!/usr/bin/env python
"""
Demo script for the HelicalHypersphericalNetwork showing how it can learn to represent
numbers as helices and perform addition using the Clock algorithm.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from HelicalHypersphericalNetwork import HelicalHypersphericalNetwork
import os

# Configure matplotlib for better visualization
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    print("== HelicalHypersphericalNetwork Demonstration ==")
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Configuration
    input_dim = 1  # Single number input
    output_dim = 32  # Dimension of the hyperspherical embedding
    helix_periods = [2, 5, 10, 100]  # Periods for helical representation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = HelicalHypersphericalNetwork(
        input_dim=input_dim, 
        output_dim=output_dim,
        num_helices=len(helix_periods),
        helix_periods=helix_periods
    ).to(device)
    
    print(f"Model initialized with periods: {helix_periods}")
    print(f"Using device: {device}")
    
    # Generate number data
    numbers = torch.arange(0, 100, 1.0).to(device)
    numbers_input = numbers.view(-1, 1)  # Reshape for model input [batch_size, input_dim]
    
    # Forward pass to get helical representations
    with torch.no_grad():
        helix_representations = model(numbers_input)
    
    print(f"Generated representations with shape: {helix_representations.shape}")
    
    # Visualize the representations for each period
    visualize_helical_representations(model, numbers, helix_representations)
    
    # Demonstrate the Clock algorithm for addition
    demonstrate_clock_addition(model, numbers)
    
    print("== Demonstration Complete ==")
    print("Results saved to results/ directory")

def visualize_helical_representations(model, numbers, representations):
    """Visualize helical representations for different periods."""
    num_periods = len(model.helix_periods)
    
    # Create PCA visualizations of the embeddings
    from sklearn.decomposition import PCA
    
    fig, axes = plt.subplots(1, num_periods, figsize=(5*num_periods, 5))
    if num_periods == 1:
        axes = [axes]
    
    for i, period in enumerate(model.helix_periods):
        # Extract representations for this period
        period_reps = representations[:, i, :].cpu().numpy()
        
        # Apply PCA to reduce to 3D
        pca = PCA(n_components=3)
        reduced_reps = pca.fit_transform(period_reps)
        
        # Create 3D subplot
        ax = axes[i]
        scatter = ax.scatter(
            reduced_reps[:, 0], 
            reduced_reps[:, 1], 
            c=numbers.cpu().numpy(), 
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
        
        # Set labels
        ax.set_title(f'Helical Representation (T={period})')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # Draw a circle to illustrate the helical structure
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'r--', alpha=0.3)
        
        # Draw modular equivalence classes
        for j in range(period):
            # Mark numbers that are equivalent modulo this period
            equiv_indices = [k for k, num in enumerate(numbers.cpu().numpy()) if int(num) % period == j]
            if equiv_indices:
                ax.scatter(
                    reduced_reps[equiv_indices, 0], 
                    reduced_reps[equiv_indices, 1],
                    edgecolor='red', 
                    facecolor='none', 
                    s=100, 
                    alpha=0.5,
                    label=f'{j} mod {period}'
                )
    
    plt.tight_layout()
    plt.savefig("results/helical_representations.png", dpi=300, bbox_inches='tight')
    print("Saved visualization of helical representations to results/helical_representations.png")
    plt.close()

def demonstrate_clock_addition(model, numbers, num_examples=5):
    """Demonstrate adding numbers using the Clock algorithm."""
    # Select a few number pairs for demonstration
    np.random.seed(42)
    pairs = []
    for _ in range(num_examples):
        a = np.random.randint(0, 50)
        b = np.random.randint(0, 50)
        pairs.append((a, b))
    
    # Create input tensors
    device = numbers.device
    a_inputs = torch.tensor([p[0] for p in pairs], dtype=torch.float32).view(-1, 1).to(device)
    b_inputs = torch.tensor([p[1] for p in pairs], dtype=torch.float32).view(-1, 1).to(device)
    
    # Get representations
    with torch.no_grad():
        a_reps = model(a_inputs)
        b_reps = model(b_inputs)
    
    # Create visualization for each period
    num_periods = len(model.helix_periods)
    
    fig, axes = plt.subplots(num_examples, num_periods, figsize=(5*num_periods, 5*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for example_idx, (a, b) in enumerate(pairs):
        result = a + b
        
        for period_idx, period in enumerate(model.helix_periods):
            ax = axes[example_idx][period_idx]
            
            # Extract 2D projections (first two components for simplicity)
            a_proj = a_reps[example_idx, period_idx, :2].cpu().numpy()
            b_proj = b_reps[example_idx, period_idx, :2].cpu().numpy()
            
            # Compute expected result using the Clock algorithm
            a_angle = 2 * np.pi * (a % period) / period
            b_angle = 2 * np.pi * (b % period) / period
            result_angle = 2 * np.pi * (result % period) / period
            
            # Draw the circle
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            ax.plot(circle_x, circle_y, 'k--', alpha=0.3)
            
            # Plot a, b and a+b
            ax.scatter([np.cos(a_angle)], [np.sin(a_angle)], color='blue', s=100, label=f'a={a}')
            ax.scatter([np.cos(b_angle)], [np.sin(b_angle)], color='green', s=100, label=f'b={b}')
            ax.scatter([np.cos(result_angle)], [np.sin(result_angle)], color='red', s=100, label=f'a+b={result}')
            
            # Draw arrow from a to a+b to illustrate rotation
            ax.arrow(
                np.cos(a_angle), np.sin(a_angle),
                np.cos(result_angle) - np.cos(a_angle),
                np.sin(result_angle) - np.sin(a_angle),
                head_width=0.05, head_length=0.1,
                fc='red', ec='red', alpha=0.7
            )
            
            ax.set_title(f'{a} + {b} = {result} (mod {period})')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            
            if example_idx == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig("results/clock_algorithm_demo.png", dpi=300, bbox_inches='tight')
    print("Saved visualization of the Clock algorithm to results/clock_algorithm_demo.png")
    plt.close()

if __name__ == "__main__":
    main() 