#!/usr/bin/env python
"""
Training script for the HelicalHypersphericalNetwork on addition tasks.

This script demonstrates how to train the network to learn helical number
representations that facilitate addition via the Clock algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from HelicalHypersphericalNetwork import HelicalHypersphericalNetwork
import os
import time
from tqdm import tqdm

def main():
    print("=== Training HelicalHypersphericalNetwork for Addition ===")
    
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Configuration
    input_dim = 1  # Single number input
    output_dim = 32  # Dimension of the hyperspherical embedding
    helix_periods = [2, 5, 10, 100]  # Periods from the paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training settings
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    max_number = 100
    
    # Initialize model
    model = HelicalHypersphericalNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_helices=len(helix_periods),
        helix_periods=helix_periods
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create datasets
    train_loader = create_addition_dataset(max_number, batch_size, train=True)
    val_loader = create_addition_dataset(max_number, batch_size, train=False)
    
    # Train the model
    train_losses, val_losses = train_model(
        model, optimizer, train_loader, val_loader, num_epochs, device
    )
    
    # Plot training progress
    plot_training_progress(train_losses, val_losses)
    
    # Save the model
    torch.save(model.state_dict(), "models/helical_network.pt")
    
    # Evaluate addition performance
    evaluate_addition(model, device, max_number)
    
    print("=== Training Complete ===")
    print("Model saved to models/helical_network.pt")
    print("Results saved to results/ directory")

def create_addition_dataset(max_number, batch_size, train=True):
    """
    Create a dataset of addition problems.
    
    For training: random pairs (a, b) and their sum
    """
    # Use different random seeds for train and validation to avoid overlap
    rng = np.random.RandomState(42 if train else 43)
    
    # Number of samples
    num_samples = 10000 if train else 2000
    
    # Generate random pairs
    a_values = rng.randint(0, max_number, num_samples)
    b_values = rng.randint(0, max_number, num_samples)
    sums = a_values + b_values
    
    # Create PyTorch tensors
    a_tensor = torch.tensor(a_values, dtype=torch.float32).view(-1, 1)
    b_tensor = torch.tensor(b_values, dtype=torch.float32).view(-1, 1)
    sum_tensor = torch.tensor(sums, dtype=torch.float32).view(-1, 1)
    
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(a_tensor, b_tensor, sum_tensor)
    
    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train
    )
    
    return loader

def train_model(model, optimizer, train_loader, val_loader, num_epochs, device):
    """Train the helical network model."""
    print(f"Training on {device} for {num_epochs} epochs...")
    
    # Track losses
    train_losses = []
    val_losses = []
    
    # MSE loss for regression
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for a, b, target_sum in progress_bar:
            # Move data to device
            a, b, target_sum = a.to(device), b.to(device), target_sum.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            a_reps = model(a)
            b_reps = model(b)
            
            # Compute combined representation using the Clock algorithm
            combined_loss = 0.0
            for i in range(len(model.helix_periods)):
                # Get the expected representation for the sum
                expected_sum_rep = model(target_sum)[:, i, :]
                
                # For the Clock algorithm, we need to extract the cosine/sine components
                # from the high-dimensional representation
                # We'll use just the first two dimensions as the paper suggests
                a_period_cossin = a_reps[:, i, :2]  # Use first 2 dimensions for cos/sin
                b_period_cossin = b_reps[:, i, :2]  # Use first 2 dimensions for cos/sin
                
                # Apply Clock algorithm (outputs 2D vector with cos/sin components)
                sum_pred_cossin = model.compute_clock_algorithm(a_period_cossin, b_period_cossin, i)
                
                # Calculate loss on just the cosine/sine components
                expected_sum_cossin = expected_sum_rep[:, :2]  # First two components
                period_loss = criterion(sum_pred_cossin, expected_sum_cossin)
                
                combined_loss += period_loss
            
            # Average loss across periods
            loss = combined_loss / len(model.helix_periods)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * a.size(0)
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for a, b, target_sum in val_loader:
                # Move data to device
                a, b, target_sum = a.to(device), b.to(device), target_sum.to(device)
                
                # Forward pass
                a_reps = model(a)
                b_reps = model(b)
                
                # Compute combined loss across periods
                combined_loss = 0.0
                for i in range(len(model.helix_periods)):
                    # Extract cosine/sine components (first 2 dimensions)
                    a_period_cossin = a_reps[:, i, :2]
                    b_period_cossin = b_reps[:, i, :2]
                    
                    # Apply Clock algorithm
                    sum_pred_cossin = model.compute_clock_algorithm(a_period_cossin, b_period_cossin, i)
                    
                    # Expected cosine/sine components
                    expected_sum_rep = model(target_sum)[:, i, :]
                    expected_sum_cossin = expected_sum_rep[:, :2]
                    
                    # Compute loss
                    period_loss = criterion(sum_pred_cossin, expected_sum_cossin)
                    combined_loss += period_loss
                
                # Average loss across periods
                loss = combined_loss / len(model.helix_periods)
                val_loss += loss.item() * a.size(0)
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses

def plot_training_progress(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/training_progress.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved training progress plot to results/training_progress.png")

def evaluate_addition(model, device, max_number, num_examples=100):
    """Evaluate the model's performance on addition tasks."""
    # Generate test examples
    rng = np.random.RandomState(44)  # New seed for test data
    a_values = rng.randint(0, max_number, num_examples)
    b_values = rng.randint(0, max_number, num_examples)
    sums = a_values + b_values
    
    # Create tensors
    a_tensor = torch.tensor(a_values, dtype=torch.float32).view(-1, 1).to(device)
    b_tensor = torch.tensor(b_values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Predict using the model
    model.eval()
    with torch.no_grad():
        a_reps = model(a_tensor)
        b_reps = model(b_tensor)
        
        # Use the best period for prediction (typically period 10 for base-10 numbers)
        best_period_idx = model.helix_periods.index(10) if 10 in model.helix_periods else 0
        
        # Compute sum representations using Clock algorithm
        # Extract just the cosine/sine components for the clock algorithm
        a_period_cossin = a_reps[:, best_period_idx, :2]
        b_period_cossin = b_reps[:, best_period_idx, :2]
        
        # Predict sums using the Clock algorithm
        predicted_sum_cossin = model.compute_clock_algorithm(
            a_period_cossin, 
            b_period_cossin, 
            best_period_idx
        )
        
        # For a proper evaluation, we'd need to decode these representations back to numbers
        # This would require additional machinery (e.g. a decoder network)
        # For this demonstration, we'll visualize the representations instead
        
        # Sample a few examples to visualize
        sample_indices = rng.choice(num_examples, size=5, replace=False)
        
        fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = [axes]
        
        for i, idx in enumerate(sample_indices):
            a = a_values[idx]
            b = b_values[idx]
            true_sum = sums[idx]
            
            # Get representations
            a_rep = a_period_cossin[idx].cpu().numpy()
            b_rep = b_period_cossin[idx].cpu().numpy()
            sum_rep = predicted_sum_cossin[idx].cpu().numpy()
            
            # Draw circle
            ax = axes[i]
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            ax.plot(circle_x, circle_y, 'k--', alpha=0.3)
            
            # Plot points
            ax.scatter([a_rep[0]], [a_rep[1]], color='blue', s=100, label=f'a={a}')
            ax.scatter([b_rep[0]], [b_rep[1]], color='green', s=100, label=f'b={b}')
            ax.scatter([sum_rep[0]], [sum_rep[1]], color='red', s=100, label=f'Predicted: a+b')
            
            # Draw angle for true sum
            period = model.helix_periods[best_period_idx]
            true_angle = 2 * np.pi * (true_sum % period) / period
            true_x, true_y = np.cos(true_angle), np.sin(true_angle)
            ax.scatter([true_x], [true_y], color='purple', s=100, marker='*', 
                      label=f'True: {a}+{b}={true_sum}')
            
            ax.set_title(f'Example {i+1}: {a} + {b} = {true_sum}')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig("results/addition_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved addition evaluation results to results/addition_evaluation.png")

if __name__ == "__main__":
    main() 