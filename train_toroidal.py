import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
from toroidal_network import ToroidalNetwork
import math

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_toroidal_data(n_samples=1000, noise_level=0.1):
    """
    Generate synthetic data with properties well-suited for toroidal representation.
    The data consists of periodic components with different frequencies.
    """
    # Generate angles uniformly distributed on circles
    theta1 = np.random.uniform(0, 2*np.pi, n_samples)
    theta2 = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Create features based on these angles
    features = []
    
    # First periodic component
    features.append(np.sin(theta1))
    features.append(np.cos(theta1))
    features.append(np.sin(2*theta1))
    features.append(np.cos(2*theta1))
    
    # Second periodic component
    features.append(np.sin(theta2))
    features.append(np.cos(theta2))
    features.append(np.sin(3*theta2))
    features.append(np.cos(3*theta2))
    
    # Interaction terms
    features.append(np.sin(theta1 + theta2))
    features.append(np.cos(theta1 + theta2))
    
    # Stack into feature matrix
    X = np.column_stack(features)
    
    # Add some noise
    X += noise_level * np.random.randn(*X.shape)
    
    # Generate classes based on regions in the torus (based on both angles)
    num_classes = 5
    class_boundaries = np.linspace(0, 2*np.pi, num_classes+1)
    
    # Determine class based on position in the torus
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Find which section of the first circle
        idx1 = np.digitize(theta1[i], class_boundaries) - 1
        if idx1 == num_classes:
            idx1 = 0
            
        # Find which section of the second circle
        idx2 = np.digitize(theta2[i], class_boundaries) - 1
        if idx2 == num_classes:
            idx2 = 0
            
        # Combine to get a class
        labels[i] = (idx1 + idx2) % num_classes
    
    # Also generate a regression target (e.g., a function of the angles)
    regression_targets = (np.sin(theta1) + np.cos(theta2)) / 2
    
    # Normalize to [0, 2π] for circular regression
    regression_targets = (regression_targets - regression_targets.min()) / (regression_targets.max() - regression_targets.min())
    regression_targets = regression_targets * 2 * np.pi
    
    return X, labels, regression_targets, theta1, theta2

def train(args):
    # Generate synthetic data
    print("Generating synthetic toroidal data...")
    X, y, regression_targets, theta1, theta2 = generate_toroidal_data(
        n_samples=args.n_samples, 
        noise_level=args.noise_level
    )
    
    # Split data
    X_train, X_test, y_train, y_test, reg_train, reg_test = train_test_split(
        X, y, regression_targets, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    reg_train = torch.FloatTensor(reg_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    reg_test = torch.FloatTensor(reg_test)
    
    # Ensure regression targets are properly shaped (batch_size,)
    if len(reg_train.shape) > 1 and reg_train.shape[1] > 1:
        reg_train = reg_train.reshape(-1)
    if len(reg_test.shape) > 1 and reg_test.shape[1] > 1:
        reg_test = reg_test.reshape(-1)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, reg_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    # Count number of classes in the dataset
    num_classes = len(torch.unique(y_train))
    print(f"Dataset contains {num_classes} classes")
    
    # Custom initialization for correct number of prototypes
    class CustomToroidalNetwork(ToroidalNetwork):
        def _initialize_prototypes(self):
            return super()._initialize_prototypes(num_classes=num_classes)
    
    # Initialize models
    input_dim = X_train.shape[1]
    toroidal_net = CustomToroidalNetwork(
        input_dim=input_dim,
        output_dim=args.output_dim,
        torus_dims=args.torus_dims,
        period_pairs=[(2, 5), (10, 100)]  # Example periods
    )
    
    # Define optimizer
    optimizer = optim.Adam(toroidal_net.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        toroidal_net.train()
        epoch_loss = 0
        
        for batch_idx, (data, target, reg_target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = toroidal_net(data)
            
            # Compute loss (combined classification and regression)
            if args.task == 'both':
                loss = toroidal_net.combined_loss(outputs, target, reg_target)
            elif args.task == 'classification':
                loss = toroidal_net.classification_loss(outputs, target)
            else:  # regression
                loss = toroidal_net.regression_loss(outputs, reg_target)
                
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on test set every few epochs
        if (epoch + 1) % 5 == 0:
            evaluate(toroidal_net, X_test, y_test, reg_test, args.task)
    
    print("Training completed!")
    
    # Final evaluation
    evaluate(toroidal_net, X_test, y_test, reg_test, args.task)
    
    # Visualize embeddings if requested
    if args.visualize:
        visualize_embeddings(toroidal_net, X_test, y_test)

def evaluate(model, X_test, y_test, reg_test, task):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        
        # Ensure regression targets have correct shape
        if len(reg_test.shape) > 1 and reg_test.shape[1] > 1:
            reg_test = reg_test.reshape(-1)
        
        if task in ['classification', 'both']:
            # Evaluate classification
            similarities = model.compute_similarity(outputs)
            _, predicted = torch.max(similarities, 1)
            accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
            print(f"Classification Accuracy: {accuracy:.4f}")
            
        if task in ['regression', 'both']:
            # Evaluate regression
            reg_loss = model.regression_loss(outputs, reg_test)
            print(f"Regression Loss: {reg_loss.item():.4f}")

def visualize_embeddings(model, X_test, y_test):
    """Visualize the toroidal embeddings."""
    model.eval()
    with torch.no_grad():
        # Get embeddings
        outputs = model(X_test)
        batch_size = outputs.size(0)
        
        # Reshape to get pairs of (cos, sin) for each circle
        outputs_reshaped = outputs.reshape(batch_size, -1, 2)  # [batch, num_circles, 2]
        
        # Convert to angles
        angles = torch.atan2(outputs_reshaped[:, :, 1], outputs_reshaped[:, :, 0])
        angles = angles.numpy()
        
        # Get the first two circles for visualization (first torus dimension)
        theta1 = angles[:, 0]
        theta2 = angles[:, 1]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(theta1, theta2, c=y_test.numpy(), cmap='viridis', alpha=0.6)
        plt.colorbar(label='Class')
        plt.xlabel('Angle 1')
        plt.ylabel('Angle 2')
        plt.title('Toroidal Embedding (First Torus Dimension)')
        plt.savefig('toroidal_embeddings.png')
        plt.close()
        
        print("Embedding visualization saved to 'toroidal_embeddings.png'")
        
        # Also visualize another pair of circles
        if angles.shape[1] >= 4:
            plt.figure(figsize=(10, 8))
            plt.scatter(angles[:, 2], angles[:, 3], c=y_test.numpy(), cmap='viridis', alpha=0.6)
            plt.colorbar(label='Class')
            plt.xlabel('Angle 3')
            plt.ylabel('Angle 4')
            plt.title('Toroidal Embedding (Second Torus Dimension)')
            plt.savefig('toroidal_embeddings_dim2.png')
            plt.close()
            
            print("Second embedding visualization saved to 'toroidal_embeddings_dim2.png'")

def main():
    parser = argparse.ArgumentParser(description='Train Toroidal Network')
    
    # Dataset parameters
    parser.add_argument('--n_samples', type=int, default=2000, 
                        help='Number of synthetic samples to generate')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise level for synthetic data')
    
    # Model parameters
    parser.add_argument('--output_dim', type=int, default=16,
                        help='Dimension of output embeddings per torus component')
    parser.add_argument('--torus_dims', type=int, default=2,
                        help='Number of torus dimensions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'both'],
                        default='both', help='Task type')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize embeddings after training')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main() 