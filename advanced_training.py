#!/usr/bin/env python
"""
Advanced training script for the HelicalHypersphericalNetwork.

This script implements a more complex training regimen to better test the 
model's capabilities and compare it with baseline architectures.
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define a simple baseline MLP for comparison
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[512, 256, 128], output_dim=1):
        super(BaselineMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Define a transformer-based model for comparison
class TransformerNumberModel(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2, output_dim=1):
        super(TransformerNumberModel, self).__init__()
        
        # Embedding layer to map input to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        # Reshape input to [batch_size, seq_len, input_dim]
        # For our case, seq_len is typically 2 (two numbers for binary operations)
        batch_size = x.size(0)
        seq_len = x.size(1) if x.dim() > 1 else 1
        x = x.view(batch_size, seq_len, -1)
        
        # Apply embedding
        x = self.embedding(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Take the first token's output for the final prediction
        x = x[:, 0, :]
        
        # Apply output layer
        return self.output_layer(x)

def main():
    print("=== Advanced Training for HelicalHypersphericalNetwork ===")
    
    # Create directories
    os.makedirs("results/advanced", exist_ok=True)
    os.makedirs("models/advanced", exist_ok=True)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training settings
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.001
    min_number = -1000  # Expanded number range
    max_number = 1000   # Expanded number range
    
    # Train on multiple tasks
    tasks = [
        {"name": "addition", "operation": lambda a, b: a + b},
        {"name": "subtraction", "operation": lambda a, b: a - b},
        {"name": "multiplication", "operation": lambda a, b: a * b},
        {"name": "squared", "operation": lambda a, b: a**2},
        {"name": "complex", "operation": lambda a, b: a**2 + 2*a*b + b**2}  # (a+b)^2
    ]
    
    # Initialize models
    models = {}
    optimizers = {}
    
    # Helical model configuration
    input_dim = 1  # Single number input
    output_dim = 64  # Increased dimension for more complex tasks
    helix_periods = [2, 3, 5, 7, 10, 20, 50, 100, 1000]  # More periods for complex patterns
    
    # Initialize our model
    models["helical"] = HelicalHypersphericalNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_helices=len(helix_periods),
        helix_periods=helix_periods
    ).to(device)
    
    # Initialize baseline MLP
    models["mlp"] = BaselineMLP(
        input_dim=2,  # Two numbers as input concatenated
        hidden_dims=[512, 256, 128],
        output_dim=1
    ).to(device)
    
    # Initialize transformer model
    models["transformer"] = TransformerNumberModel(
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=3,
        output_dim=1
    ).to(device)
    
    # Create optimizers
    for model_name, model in models.items():
        optimizers[model_name] = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Results tracking
    results = {task["name"]: {model_name: {"train_loss": [], "val_loss": [], "test_metrics": {}} 
                             for model_name in models.keys()} 
              for task in tasks}
    
    # Train on each task
    for task in tasks:
        task_name = task["name"]
        operation = task["operation"]
        
        print(f"\n=== Training on task: {task_name} ===")
        
        # Create datasets
        train_loader = create_dataset(min_number, max_number, batch_size, operation, train=True)
        val_loader = create_dataset(min_number, max_number, batch_size, operation, train=False)
        
        # Train each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name} model on {task_name}...")
            
            # Train the model
            train_losses, val_losses = train_model_on_task(
                model, 
                optimizers[model_name], 
                train_loader, 
                val_loader, 
                num_epochs, 
                device,
                model_name
            )
            
            # Store losses
            results[task_name][model_name]["train_loss"] = train_losses
            results[task_name][model_name]["val_loss"] = val_losses
            
            # Save the model
            torch.save(model.state_dict(), f"models/advanced/{model_name}_{task_name}.pt")
            
            # Evaluate on test data
            test_metrics = evaluate_model(model, device, min_number, max_number, operation, model_name)
            results[task_name][model_name]["test_metrics"] = test_metrics
    
    # Plot comparative results
    plot_comparative_results(results, tasks)
    
    print("\n=== Advanced Training Complete ===")
    print("Models saved to models/advanced/ directory")
    print("Results saved to results/advanced/ directory")

def create_dataset(min_number, max_number, batch_size, operation, train=True):
    """Create a dataset for the specified operation."""
    # Use different random seeds for train and validation to avoid overlap
    rng = np.random.RandomState(42 if train else 43)
    
    # Number of samples
    num_samples = 20000 if train else 4000
    
    # Generate random values
    a_values = rng.uniform(min_number, max_number, num_samples)
    b_values = rng.uniform(min_number, max_number, num_samples)
    
    # Apply the operation
    if operation.__code__.co_argcount == 2:  # Binary operation
        results = operation(a_values, b_values)
    else:  # Unary operation (only uses a_values)
        results = operation(a_values, None)
    
    # Create PyTorch tensors
    a_tensor = torch.tensor(a_values, dtype=torch.float32).view(-1, 1)
    b_tensor = torch.tensor(b_values, dtype=torch.float32).view(-1, 1)
    result_tensor = torch.tensor(results, dtype=torch.float32).view(-1, 1)
    
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(a_tensor, b_tensor, result_tensor)
    
    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train
    )
    
    return loader

def train_model_on_task(model, optimizer, train_loader, val_loader, num_epochs, device, model_name):
    """Train the specified model on a given task."""
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
        for a, b, target in progress_bar:
            # Move data to device
            a, b, target = a.to(device), b.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - depends on model type
            if model_name == "helical":
                # Helical model processes each number separately then combines them
                a_reps = model(a)
                b_reps = model(b)
                
                # Combine using the clock algorithm for all periods
                combined_loss = 0.0
                for i in range(len(model.helix_periods)):
                    a_period_cossin = a_reps[:, i, :2]
                    b_period_cossin = b_reps[:, i, :2]
                    
                    # Apply Clock algorithm 
                    sum_pred_cossin = model.compute_clock_algorithm(a_period_cossin, b_period_cossin, i)
                    
                    # For now, we're still using the same loss as in the original training
                    # In a more advanced implementation, we'd have a decoder to map from 
                    # representations back to numbers
                    with torch.no_grad():
                        expected_rep = model(target)[:, i, :]
                    
                    expected_cossin = expected_rep[:, :2]
                    period_loss = criterion(sum_pred_cossin, expected_cossin)
                    combined_loss += period_loss
                
                # Average loss across periods
                loss = combined_loss / len(model.helix_periods)
            
            elif model_name == "mlp":
                # MLP takes concatenated inputs
                inputs = torch.cat((a, b), dim=1)
                outputs = model(inputs)
                loss = criterion(outputs, target)
            
            elif model_name == "transformer":
                # Transformer takes sequence of inputs
                inputs = torch.stack((a.squeeze(), b.squeeze()), dim=1).unsqueeze(-1)
                outputs = model(inputs)
                loss = criterion(outputs, target)
            
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
            for a, b, target in val_loader:
                # Move data to device
                a, b, target = a.to(device), b.to(device), target.to(device)
                
                # Forward pass - depends on model type
                if model_name == "helical":
                    # Helical model
                    a_reps = model(a)
                    b_reps = model(b)
                    
                    # Combine using the clock algorithm
                    combined_loss = 0.0
                    for i in range(len(model.helix_periods)):
                        a_period_cossin = a_reps[:, i, :2]
                        b_period_cossin = b_reps[:, i, :2]
                        
                        # Apply Clock algorithm
                        sum_pred_cossin = model.compute_clock_algorithm(a_period_cossin, b_period_cossin, i)
                        
                        expected_rep = model(target)[:, i, :]
                        expected_cossin = expected_rep[:, :2]
                        
                        period_loss = criterion(sum_pred_cossin, expected_cossin)
                        combined_loss += period_loss
                    
                    # Average loss across periods
                    loss = combined_loss / len(model.helix_periods)
                
                elif model_name == "mlp":
                    # MLP
                    inputs = torch.cat((a, b), dim=1)
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                
                elif model_name == "transformer":
                    # Transformer
                    inputs = torch.stack((a.squeeze(), b.squeeze()), dim=1).unsqueeze(-1)
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                
                # Update statistics
                val_loss += loss.item() * a.size(0)
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, device, min_number, max_number, operation, model_name, num_examples=1000):
    """Evaluate the model's performance on test data."""
    print(f"Evaluating {model_name} model...")
    
    # Generate test examples
    rng = np.random.RandomState(44)  # New seed for test data
    a_values = rng.uniform(min_number, max_number, num_examples)
    b_values = rng.uniform(min_number, max_number, num_examples)
    
    # Apply operation to get true results
    if operation.__code__.co_argcount == 2:  # Binary operation
        true_results = operation(a_values, b_values)
    else:  # Unary operation
        true_results = operation(a_values, None)
    
    # Create tensors
    a_tensor = torch.tensor(a_values, dtype=torch.float32).view(-1, 1).to(device)
    b_tensor = torch.tensor(b_values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Predict using the model
    model.eval()
    predicted_results = []
    
    with torch.no_grad():
        # Make predictions in batches
        batch_size = 100
        num_batches = num_examples // batch_size + (1 if num_examples % batch_size > 0 else 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_examples)
            
            batch_a = a_tensor[start_idx:end_idx]
            batch_b = b_tensor[start_idx:end_idx]
            
            if model_name == "helical":
                # For the helical model, we need to decode from the representations
                # This is an approximate method - in a real implementation, we'd have a trained decoder
                a_reps = model(batch_a)
                b_reps = model(batch_b)
                
                # Use the period 10 as it's good for decimal numbers
                best_period_idx = model.helix_periods.index(10) if 10 in model.helix_periods else 0
                
                a_cossin = a_reps[:, best_period_idx, :2]
                b_cossin = b_reps[:, best_period_idx, :2]
                
                # Use the Clock algorithm
                result_cossin = model.compute_clock_algorithm(a_cossin, b_cossin, best_period_idx)
                
                # Convert to angles (approximate decoding)
                angles = torch.atan2(result_cossin[:, 1], result_cossin[:, 0])
                
                # Map angles back to values (this is very approximate)
                # A better approach would be to train a separate decoder network
                period = model.helix_periods[best_period_idx]
                batch_predictions = (angles / (2 * np.pi) * period) % period
                
                # For operations other than addition, this naive decoding won't work well
                # This is a limitation of the current implementation
                
            elif model_name == "mlp":
                inputs = torch.cat((batch_a, batch_b), dim=1)
                batch_predictions = model(inputs).squeeze()
            
            elif model_name == "transformer":
                inputs = torch.stack((batch_a.squeeze(), batch_b.squeeze()), dim=1).unsqueeze(-1)
                batch_predictions = model(inputs).squeeze()
            
            predicted_results.append(batch_predictions.cpu().numpy())
    
    # Combine batch predictions
    predicted_results = np.concatenate(predicted_results)
    
    # Compute metrics
    mse = mean_squared_error(true_results, predicted_results)
    mae = mean_absolute_error(true_results, predicted_results)
    r2 = r2_score(true_results, predicted_results)
    
    # Print results
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

def plot_comparative_results(results, tasks):
    """Plot comparative results across models and tasks."""
    # Create a directory for the plots
    os.makedirs("results/advanced/comparison", exist_ok=True)
    
    # Plot training curves for each task
    for task in tasks:
        task_name = task["name"]
        
        # Training loss plot
        plt.figure(figsize=(12, 6))
        
        for model_name in results[task_name].keys():
            train_losses = results[task_name][model_name]["train_loss"]
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, label=f"{model_name} model")
        
        plt.title(f'Training Loss Comparison - {task_name.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale often better for loss comparisons
        plt.savefig(f"results/advanced/comparison/{task_name}_training_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    # Plot comparative bar charts for test metrics
    metrics = ["mse", "mae", "r2"]
    metric_names = {"mse": "Mean Squared Error", "mae": "Mean Absolute Error", "r2": "R² Score"}
    
    for metric in metrics:
        plt.figure(figsize=(14, 7))
        
        # Prepare data
        task_names = [task["name"] for task in tasks]
        model_names = list(results[task_names[0]].keys())
        
        # For each model, get its performance across all tasks
        bar_width = 0.2
        index = np.arange(len(task_names))
        
        for i, model_name in enumerate(model_names):
            model_performance = [results[task_name][model_name]["test_metrics"][metric] for task_name in task_names]
            
            # Adjust for R² which is better when higher
            if metric == "r2":
                plt.bar(index + i * bar_width, model_performance, bar_width, label=model_name)
            else:
                plt.bar(index + i * bar_width, model_performance, bar_width, label=model_name)
        
        plt.xlabel('Task')
        plt.ylabel(metric_names[metric])
        plt.title(f'Model Comparison - {metric_names[metric]}')
        plt.xticks(index + bar_width, [task_name.capitalize() for task_name in task_names])
        plt.legend()
        
        # Adjust y-axis for R² which is better when closer to 1
        if metric == "r2":
            plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"results/advanced/comparison/{metric}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a summary plot
    plt.figure(figsize=(15, 10))
    
    # Prepare data for radar chart
    task_names = [task["name"] for task in tasks]
    model_names = list(results[task_names[0]].keys())
    
    # Normalize metrics for radar chart (higher is better)
    normalized_results = {}
    
    for task_name in task_names:
        normalized_results[task_name] = {}
        
        # Get min and max for each metric across all models
        metric_ranges = {}
        for metric in ["mse", "mae"]:  # For these, lower is better
            values = [results[task_name][model_name]["test_metrics"][metric] for model_name in model_names]
            metric_ranges[metric] = (min(values), max(values))
        
        # For R², higher is better (no normalization needed)
        
        # Normalize
        for model_name in model_names:
            normalized_results[task_name][model_name] = {}
            
            for metric in ["mse", "mae"]:
                min_val, max_val = metric_ranges[metric]
                val = results[task_name][model_name]["test_metrics"][metric]
                
                if max_val == min_val:
                    # All models performed the same
                    normalized_results[task_name][model_name][metric] = 1.0
                else:
                    # Invert so that lower error = higher score
                    normalized_results[task_name][model_name][metric] = 1 - (val - min_val) / (max_val - min_val)
            
            # For R², higher is better (values typically between 0 and 1)
            normalized_results[task_name][model_name]["r2"] = results[task_name][model_name]["test_metrics"]["r2"]
    
    # Save the normalized results to a CSV file
    with open("results/advanced/comparison/normalized_results.csv", "w") as f:
        # Write header
        f.write("Task,Model,MSE_normalized,MAE_normalized,R2\n")
        
        for task_name in task_names:
            for model_name in model_names:
                f.write(f"{task_name},{model_name},")
                f.write(f"{normalized_results[task_name][model_name]['mse']:.4f},")
                f.write(f"{normalized_results[task_name][model_name]['mae']:.4f},")
                f.write(f"{normalized_results[task_name][model_name]['r2']:.4f}\n")
    
    print("Saved normalized results to results/advanced/comparison/normalized_results.csv")
    
    # Calculate average performance across all tasks
    plt.figure(figsize=(10, 6))
    
    # Compute averages
    avg_performance = {model_name: [] for model_name in model_names}
    
    for metric in ["mse", "mae", "r2"]:
        for model_name in model_names:
            # For MSE and MAE, higher normalized score is better
            if metric in ["mse", "mae"]:
                avg = np.mean([normalized_results[task_name][model_name][metric] for task_name in task_names])
            else:  # For R², higher raw score is better
                avg = np.mean([results[task_name][model_name]["test_metrics"][metric] for task_name in task_names])
            
            avg_performance[model_name].append(avg)
    
    # Create grouped bar chart
    x = np.arange(3)  # 3 metrics
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        plt.bar(x + i*width, avg_performance[model_name], width, label=model_name)
    
    plt.xlabel('Metric')
    plt.ylabel('Average Performance (higher is better)')
    plt.title('Overall Model Performance Across All Tasks')
    plt.xticks(x + width, ['MSE (normalized)', 'MAE (normalized)', 'R²'])
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"results/advanced/comparison/overall_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 