import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import argparse
import time
from toroidal_network import ToroidalNetwork
import sys
import os

# Add the helical network from the original file
class HelicalHypersphericalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_helices=4, helix_periods=[2, 5, 10, 100]):
        super(HelicalHypersphericalNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_helices = num_helices
        self.helix_periods = helix_periods
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Separate projection heads for each helix
        self.helix_projections = nn.ModuleList([
            nn.Linear(128, output_dim) for _ in range(num_helices)
        ])
        
        # Initialize prototype points uniformly on hypersphere
        self.prototypes = self._initialize_prototypes()
        
    def _initialize_prototypes(self, num_classes=10):
        """Initialize prototypes uniformly on the hypersphere using optimization."""
        # Generate initial random prototypes
        num_prototypes = num_classes
        prototypes = torch.randn(num_prototypes, self.output_dim)
        prototypes = nn.functional.normalize(prototypes, p=2, dim=1)
        
        # Use as fixed prototypes
        prototypes = nn.Parameter(prototypes, requires_grad=False)
        return prototypes
    
    def forward(self, x):
        """Forward pass with multiple helical representations."""
        # Extract features
        features = self.feature_net(x)
        
        # Project to multiple helical manifolds
        helix_outputs = []
        for i, period in enumerate(self.helix_periods):
            projection = self.helix_projections[i](features)
            # Normalize to lie on hypersphere
            projection = nn.functional.normalize(projection, p=2, dim=1)
            helix_outputs.append(projection)
        
        # Combine helical representations
        combined = torch.stack(helix_outputs, dim=1)
        
        return combined
    
    def compute_similarity(self, outputs):
        """Compute cosine similarity to all prototypes."""
        # Reshape for similarity computation
        batch_size = outputs.size(0)
        num_helices = outputs.size(1)
        output_dim = outputs.size(2)
        
        # Average across helices
        outputs = torch.mean(outputs, dim=1)
        
        # Ensure normalized
        outputs = nn.functional.normalize(outputs, p=2, dim=1)
        
        # Compute cosine similarity to all prototypes
        similarities = torch.mm(outputs, self.prototypes.t())
        
        return similarities
    
    def classification_loss(self, outputs, targets):
        """Classification loss based on prototype similarity."""
        similarities = self.compute_similarity(outputs)
        
        # Cross entropy loss
        loss = nn.functional.cross_entropy(similarities, targets)
        
        return loss
    
    def regression_loss(self, outputs, values):
        """Simple regression loss using one of the helices."""
        # Use the first helix for regression
        helix_outputs = outputs[:, 0, :]
        
        # Normalize values to [-1, 1]
        norm_values = 2 * (values - values.min()) / (values.max() - values.min()) - 1
        norm_values = norm_values.unsqueeze(1).expand(-1, helix_outputs.size(1))
        
        # Mean squared error loss
        loss = torch.mean((helix_outputs - norm_values) ** 2)
        
        return loss
    
    def combined_loss(self, outputs, targets=None, regression_values=None):
        """Combined loss for classification and/or regression."""
        loss = 0
        
        if targets is not None:
            # Add classification loss
            class_loss = self.classification_loss(outputs, targets)
            loss += class_loss
            
        if regression_values is not None:
            # Add regression loss
            reg_loss = self.regression_loss(outputs, regression_values)
            loss += reg_loss
            
        return loss

# Functions for data generation
def generate_multiperiodic_data(n_samples=1000, noise_level=0.1):
    """
    Generate synthetic data with multiple periodic components.
    This data will have structure well-suited for both toroidal and helical networks.
    """
    # Generate primary angles
    theta1 = np.random.uniform(0, 2*np.pi, n_samples)
    theta2 = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Create features based on circular and helical patterns
    features = []
    
    # Circular features (ideal for torus)
    features.append(np.sin(theta1))
    features.append(np.cos(theta1))
    features.append(np.sin(theta2))
    features.append(np.cos(theta2))
    
    # Spiral/helical features (ideal for helical manifold)
    t = np.random.uniform(0, 5, n_samples)  # helical parameter
    features.append(np.sin(t) * np.cos(theta1))
    features.append(np.sin(t) * np.sin(theta1))
    features.append(t * np.cos(theta2))
    features.append(t * np.sin(theta2))
    
    # Mixed periodic features
    for freq in [2, 3, 5]:
        features.append(np.sin(freq * theta1))
        features.append(np.cos(freq * theta2))
    
    # Stack into feature matrix
    X = np.column_stack(features)
    
    # Add some noise
    X += noise_level * np.random.randn(*X.shape)
    
    # Generate classes based on regions in the parameter space
    num_classes = 5
    class_boundaries = np.linspace(0, 2*np.pi, num_classes+1)
    
    # Determine class based on position in parameter space
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Combined classification rule
        idx1 = np.digitize(theta1[i], class_boundaries) - 1
        idx2 = np.digitize(theta2[i], class_boundaries) - 1
        if idx1 == num_classes:
            idx1 = 0
        if idx2 == num_classes:
            idx2 = 0
            
        labels[i] = (idx1 + idx2) % num_classes
    
    # Generate regression targets as a function of the parameters
    regression_targets = np.sin(theta1) * np.cos(theta2) + 0.2 * t
    
    return X, labels, regression_targets

def train_model(model_name, model, X_train, y_train, reg_train, X_test, y_test, reg_test, args):
    """Train a model and evaluate performance."""
    print(f"\nTraining {model_name}...")
    
    # Convert to PyTorch tensors if they're not already
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.LongTensor(y_train)
    if not isinstance(reg_train, torch.Tensor):
        reg_train = torch.FloatTensor(reg_train)
    
    # Ensure regression targets are properly shaped
    if len(reg_train.shape) > 1 and reg_train.shape[1] > 1:
        reg_train = reg_train.reshape(-1)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        X_train, y_train, reg_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    start_time = time.time()
    train_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target, reg_target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Ensure regression targets have the right shape
            if len(reg_target.shape) > 1 and reg_target.shape[1] > 1:
                reg_target = reg_target.reshape(-1)
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            if args.task == 'both':
                if model_name == 'Toroidal Network':
                    loss = model.combined_loss(outputs, target, reg_target)
                else:  # Helical Network
                    loss = model.combined_loss(outputs, target, reg_target)
            elif args.task == 'classification':
                loss = model.classification_loss(outputs, target)
            else:  # regression
                if model_name == 'Toroidal Network':
                    loss = model.regression_loss(outputs, reg_target)
                else:  # Helical Network
                    loss = model.regression_loss(outputs, reg_target)
                
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")
    
    # Final evaluation
    results = evaluate_model(model_name, model, X_test, y_test, reg_test, args.task)
    results['training_time'] = training_time
    results['final_train_loss'] = train_losses[-1]
    results['train_losses'] = train_losses
    
    return results

def evaluate_model(model_name, model, X_test, y_test, reg_test, task):
    """Evaluate model performance and return metrics."""
    print(f"Evaluating {model_name}...")
    model.eval()
    results = {'model_name': model_name}
    
    with torch.no_grad():
        # Convert to PyTorch tensors if needed
        if not isinstance(X_test, torch.Tensor):
            X_test_tensor = torch.FloatTensor(X_test)
        else:
            X_test_tensor = X_test
            
        if not isinstance(y_test, torch.Tensor):
            y_test_tensor = torch.LongTensor(y_test)
        else:
            y_test_tensor = y_test
            
        if not isinstance(reg_test, torch.Tensor):
            reg_test_tensor = torch.FloatTensor(reg_test)
        else:
            reg_test_tensor = reg_test
        
        # Ensure regression targets have the right shape
        if len(reg_test_tensor.shape) > 1 and reg_test_tensor.shape[1] > 1:
            reg_test_tensor = reg_test_tensor.reshape(-1)
        
        # Forward pass
        outputs = model(X_test_tensor)
        
        if task in ['classification', 'both']:
            # Evaluate classification
            similarities = model.compute_similarity(outputs)
            _, predicted = torch.max(similarities, 1)
            accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
            print(f"Classification Accuracy: {accuracy:.4f}")
            results['accuracy'] = accuracy
            
        if task in ['regression', 'both']:
            # Evaluate regression
            if model_name == 'Toroidal Network':
                reg_loss = model.regression_loss(outputs, reg_test_tensor)
                reg_mse = reg_loss.item()
            else:  # Helical Network
                reg_loss = model.regression_loss(outputs, reg_test_tensor)
                reg_mse = reg_loss.item()
                
            print(f"Regression MSE: {reg_mse:.4f}")
            results['regression_mse'] = reg_mse
    
    return results

def plot_comparison_results(results_list, save_path='comparison_results.png'):
    """Create visualizations comparing model performance."""
    # Extract results
    model_names = [r['model_name'] for r in results_list]
    accuracies = [r.get('accuracy', 0) for r in results_list]
    regression_mses = [r.get('regression_mse', 0) for r in results_list]
    training_times = [r.get('training_time', 0) for r in results_list]
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot classification accuracy
    axs[0].bar(model_names, accuracies, color=['blue', 'orange'])
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Classification Performance')
    axs[0].set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        axs[0].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Plot regression MSE
    axs[1].bar(model_names, regression_mses, color=['blue', 'orange'])
    axs[1].set_ylabel('Mean Squared Error')
    axs[1].set_title('Regression Performance')
    for i, v in enumerate(regression_mses):
        axs[1].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Plot training time
    axs[2].bar(model_names, training_times, color=['blue', 'orange'])
    axs[2].set_ylabel('Time (seconds)')
    axs[2].set_title('Training Time')
    for i, v in enumerate(training_times):
        axs[2].text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Comparison results saved to {save_path}")
    
    # Also plot training loss curves
    plt.figure(figsize=(10, 6))
    for result in results_list:
        plt.plot(result['train_losses'], label=result['model_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.savefig('training_loss_curves.png')
    plt.close()
    
    print("Training loss curves saved to training_loss_curves.png")

def main():
    parser = argparse.ArgumentParser(description='Compare Toroidal vs Helical Networks')
    
    # Dataset parameters
    parser.add_argument('--n_samples', type=int, default=2000, 
                        help='Number of synthetic samples to generate')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise level for synthetic data')
    
    # Model parameters
    parser.add_argument('--output_dim', type=int, default=16,
                        help='Dimension of output embeddings')
    parser.add_argument('--torus_dims', type=int, default=2,
                        help='Number of torus dimensions')
    parser.add_argument('--helix_periods', type=int, nargs='+', default=[2, 5, 10, 100],
                        help='Periods for helical network')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes in the dataset')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'both'],
                        default='both', help='Task type')
    
    args = parser.parse_args()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y, regression_targets = generate_multiperiodic_data(
        n_samples=args.n_samples, 
        noise_level=args.noise_level
    )
    
    # Split data
    X_train, X_test, y_train, y_test, reg_train, reg_test = train_test_split(
        X, y, regression_targets, test_size=0.2, random_state=42
    )
    
    # Ensure regression targets are properly shaped
    if len(reg_train.shape) > 1 and reg_train.shape[1] > 1:
        reg_train = reg_train.reshape(-1)
    if len(reg_test.shape) > 1 and reg_test.shape[1] > 1:
        reg_test = reg_test.reshape(-1)
    
    # Initialize models
    input_dim = X_train.shape[1]
    
    # Custom initialization function for toroidal network
    class CustomToroidalNetwork(ToroidalNetwork):
        def _initialize_prototypes(self):
            return super()._initialize_prototypes(num_classes=args.num_classes)
    
    # Custom initialization function for helical network
    class CustomHelicalNetwork(HelicalHypersphericalNetwork):
        def _initialize_prototypes(self):
            return super()._initialize_prototypes(num_classes=args.num_classes)
    
    # Initialize Toroidal Network
    toroidal_net = CustomToroidalNetwork(
        input_dim=input_dim,
        output_dim=args.output_dim,
        torus_dims=args.torus_dims,
        period_pairs=[(2, 5), (10, 100)]  # Example periods
    )
    
    # Initialize Helical Hyperspherical Network
    helical_net = CustomHelicalNetwork(
        input_dim=input_dim,
        output_dim=args.output_dim,
        num_helices=len(args.helix_periods),
        helix_periods=args.helix_periods
    )
    
    # Train and evaluate models
    toroidal_results = train_model(
        'Toroidal Network', toroidal_net, 
        X_train, y_train, reg_train, 
        X_test, y_test, reg_test, args
    )
    
    helical_results = train_model(
        'Helical Network', helical_net, 
        X_train, y_train, reg_train, 
        X_test, y_test, reg_test, args
    )
    
    # Compare and visualize results
    plot_comparison_results([toroidal_results, helical_results])
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Dataset: {args.n_samples} samples with noise level {args.noise_level}")
    print(f"Task: {args.task}")
    print("\nResults:")
    print(f"Toroidal Network - Accuracy: {toroidal_results.get('accuracy', 'N/A'):.4f}, " +
          f"Regression MSE: {toroidal_results.get('regression_mse', 'N/A'):.4f}, " +
          f"Training Time: {toroidal_results.get('training_time', 'N/A'):.2f}s")
    print(f"Helical Network - Accuracy: {helical_results.get('accuracy', 'N/A'):.4f}, " +
          f"Regression MSE: {helical_results.get('regression_mse', 'N/A'):.4f}, " +
          f"Training Time: {helical_results.get('training_time', 'N/A'):.2f}s")

if __name__ == '__main__':
    main() 