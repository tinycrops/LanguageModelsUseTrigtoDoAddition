#!/usr/bin/env python
"""
Demo script for the EnhancedHelicalNetwork showing how it can perform
multiple operations using helical representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from EnhancedHelicalNetwork import EnhancedHelicalNetwork
import os

# Configure matplotlib for better visualization
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    print("== EnhancedHelicalNetwork Demonstration ==")
    
    # Create output directory
    os.makedirs("results/enhanced", exist_ok=True)
    
    # Configuration
    input_dim = 1
    output_dim = 64
    helix_periods = [2, 3, 5, 7, 10, 20, 50, 100, 1000]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = EnhancedHelicalNetwork(
        input_dim=input_dim, 
        output_dim=output_dim,
        num_helices=len(helix_periods),
        helix_periods=helix_periods,
        learnable_periods=True
    ).to(device)
    
    print(f"Model initialized with periods: {helix_periods}")
    print(f"Using device: {device}")
    
    # Generate number data
    numbers = torch.arange(0, 100, 1.0).to(device)
    numbers_input = numbers.view(-1, 1)  # Reshape for model input [batch_size, input_dim]
    
    # Operations to demonstrate
    operations = ["addition", "subtraction", "multiplication", "squared", "complex"]
    
    # Demonstrate each operation
    for operation in operations:
        print(f"\nDemonstrating {operation} operation:")
        
        # Forward pass to get helical representations
        with torch.no_grad():
            # Get operation-specific representations
            representations = model(numbers_input, operation)
            
            print(f"  Generated {operation} representations with shape: {representations.shape}")
        
        # Visualize the representations for this operation
        visualize_operation_representations(model, numbers, representations, operation)
        
        # Demonstrate the operation
        demonstrate_operation(model, numbers, operation)
    
    print("\n== Demonstration Complete ==")
    print("Results saved to results/enhanced/ directory")

def visualize_operation_representations(model, numbers, representations, operation):
    """Visualize helical representations for different periods for a specific operation."""
    num_periods = len(model.helix_periods)
    periods_to_show = min(4, num_periods)  # Show at most 4 periods
    
    # Choose interesting periods to visualize
    period_indices = [0, 2, 4, 8][:periods_to_show]  # First, third, fifth, and ninth periods if available
    
    # Create PCA visualizations of the embeddings
    from sklearn.decomposition import PCA
    
    fig, axes = plt.subplots(1, periods_to_show, figsize=(5*periods_to_show, 5))
    if periods_to_show == 1:
        axes = [axes]
    
    for i, period_idx in enumerate(period_indices):
        period = model.helix_periods[period_idx]
        
        # Extract representations for this period
        period_reps = representations[:, period_idx, :].cpu().numpy()
        
        # Apply PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        reduced_reps = pca.fit_transform(period_reps)
        
        # Create subplot
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
        ax.set_title(f'{operation.capitalize()} - Period {period}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # Draw a circle to illustrate the helical structure
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'r--', alpha=0.3)
        
        # Draw modular equivalence classes for addition/subtraction operations
        if operation in ["addition", "subtraction"]:
            for j in range(min(period, 5)):  # Show at most 5 equivalence classes
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
    plt.savefig(f"results/enhanced/{operation}_representations.png", dpi=300, bbox_inches='tight')
    print(f"  Saved visualization of {operation} representations to results/enhanced/{operation}_representations.png")
    plt.close()

def demonstrate_operation(model, numbers, operation, num_examples=3):
    """Demonstrate an operation using the model."""
    # Select a few number pairs for demonstration
    np.random.seed(42 + operations.index(operation))  # Different seed for each operation
    
    # Range depends on operation (smaller numbers for multiplication)
    if operation in ["multiplication", "squared", "complex"]:
        max_val = 10
    else:
        max_val = 50
    
    # Generate pairs
    pairs = []
    for _ in range(num_examples):
        a = np.random.randint(0, max_val)
        b = np.random.randint(0, max_val) if operation != "squared" else 0  # For "squared", b is unused
        pairs.append((a, b))
    
    # Create input tensors
    device = numbers.device
    a_inputs = torch.tensor([p[0] for p in pairs], dtype=torch.float32).view(-1, 1).to(device)
    b_inputs = torch.tensor([p[1] for p in pairs], dtype=torch.float32).view(-1, 1).to(device)
    
    # Get period-specific results
    period_idx = model.helix_periods.index(10) if 10 in model.helix_periods else 0
    
    # Get representations
    with torch.no_grad():
        a_reps = model(a_inputs, operation)
        b_reps = model(b_inputs, operation) if operation != "squared" else None
        
        # Generate expected outputs for each operation
        if operation == "addition":
            expected_results = a_inputs + b_inputs
        elif operation == "subtraction":
            expected_results = a_inputs - b_inputs
        elif operation == "multiplication":
            expected_results = a_inputs * b_inputs
        elif operation == "squared":
            expected_results = a_inputs ** 2
        elif operation == "complex":
            # (a+b)^2 = a^2 + 2ab + b^2
            expected_results = a_inputs ** 2 + 2 * a_inputs * b_inputs + b_inputs ** 2
        
        # Get model predictions using end-to-end function
        if operation == "squared":
            # For squared, use a as both inputs
            predicted_results = model.forward_with_decode(a_inputs, a_inputs, operation)
        else:
            predicted_results = model.forward_with_decode(a_inputs, b_inputs, operation)
    
    # Create visualization of the operation
    fig, axes = plt.subplots(1, num_examples, figsize=(5*num_examples, 5))
    if num_examples == 1:
        axes = [axes]
    
    for i, (a, b) in enumerate(pairs):
        ax = axes[i]
        
        # Compute true result
        if operation == "addition":
            result = a + b
        elif operation == "subtraction":
            result = a - b
        elif operation == "multiplication":
            result = a * b
        elif operation == "squared":
            result = a ** 2
        elif operation == "complex":
            result = a ** 2 + 2 * a * b + b ** 2
        
        # Get model prediction
        prediction = predicted_results[i].item()
        
        # Extract representations for visualization
        a_period = a_reps[i, period_idx, :2].cpu().numpy()
        if operation != "squared":
            b_period = b_reps[i, period_idx, :2].cpu().numpy()
        
        # Draw the circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', alpha=0.3)
        
        # Plot points
        ax.scatter([a_period[0]], [a_period[1]], color='blue', s=100, label=f'a={a}')
        
        if operation != "squared":
            ax.scatter([b_period[0]], [b_period[1]], color='green', s=100, label=f'b={b}')
            
        # Display operation and result
        if operation == "addition":
            op_text = f"{a} + {b} = {result}"
        elif operation == "subtraction":
            op_text = f"{a} - {b} = {result}"
        elif operation == "multiplication":
            op_text = f"{a} × {b} = {result}"
        elif operation == "squared":
            op_text = f"{a}² = {result}"
        elif operation == "complex":
            op_text = f"({a} + {b})² = {result}"
        
        # Show prediction
        pred_text = f"Model: {prediction:.2f}"
        
        ax.set_title(f'{op_text}\n{pred_text}')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/enhanced/{operation}_demo.png", dpi=300, bbox_inches='tight')
    print(f"  Saved demonstration of {operation} to results/enhanced/{operation}_demo.png")
    plt.close()

if __name__ == "__main__":
    operations = ["addition", "subtraction", "multiplication", "squared", "complex"]
    main() 