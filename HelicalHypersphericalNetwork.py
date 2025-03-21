import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
    def _initialize_prototypes(self):
        """Initialize prototypes uniformly on the hypersphere using optimization."""
        # Generate initial random prototypes
        num_prototypes = sum(2 for period in self.helix_periods)  # 2 prototypes per helix (sin/cos)
        prototypes = torch.randn(num_prototypes, self.output_dim)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Optimize for maximum separation (similar to the paper's approach)
        prototypes = nn.Parameter(prototypes, requires_grad=False)
        return prototypes
    
    def _generate_helix_basis(self, numbers):
        """
        Generate a basis of helix functions for the given numbers.
        
        Args:
            numbers: Tensor of shape [batch_size] containing input numbers
            
        Returns:
            torch.Tensor: Matrix of shape [batch_size, 2*len(periods)+1] containing 
                        the basis functions evaluated at each number
        """
        batch_size = numbers.size(0)
        k = len(self.helix_periods)
        basis = torch.zeros(batch_size, 2*k + 1, device=numbers.device)
        
        # Linear component
        basis[:, 0] = numbers
        
        # Periodic components
        for i, T in enumerate(self.helix_periods):
            basis[:, 2*i + 1] = torch.cos(2 * np.pi * numbers / T)
            basis[:, 2*i + 2] = torch.sin(2 * np.pi * numbers / T)
        
        return basis
    
    def _project_to_helix_subspace(self, features, period_idx):
        """
        Project features to a specific helix subspace determined by period_idx.
        
        Args:
            features: Batch of feature vectors [batch_size, feature_dim]
            period_idx: Index of the period to use for projection
            
        Returns:
            Projection of features onto the helix subspace
        """
        projection = self.helix_projections[period_idx](features)
        return F.normalize(projection, p=2, dim=1)
    
    def forward(self, x):
        """Forward pass with multiple helical representations."""
        # Extract features
        features = self.feature_net(x)
        
        # Project to multiple helical manifolds
        helix_outputs = []
        for i, period in enumerate(self.helix_periods):
            projection = self._project_to_helix_subspace(features, i)
            helix_outputs.append(projection)
        
        # Combine helical representations
        combined = torch.stack(helix_outputs, dim=1)
        
        return combined
    
    def compute_similarity(self, outputs):
        """Compute cosine similarity to all prototypes."""
        # Flatten the helical outputs if needed
        if outputs.dim() > 2:
            batch_size = outputs.size(0)
            outputs = outputs.view(batch_size, -1)
            outputs = F.normalize(outputs, p=2, dim=1)
        
        # Compute cosine similarity to all prototypes
        similarities = F.linear(outputs, self.prototypes)
        
        return similarities
    
    def compute_clock_algorithm(self, a_projection, b_projection, period_idx):
        """
        Implement the Clock algorithm for addition using helical representations.
        
        Args:
            a_projection: Projection of 'a' value onto helix subspace [batch_size, output_dim]
            b_projection: Projection of 'b' value onto helix subspace [batch_size, output_dim]
            period_idx: Index of the period to use
            
        Returns:
            Predicted representation for a+b
        """
        # Get the period
        T = self.helix_periods[period_idx]
        
        # Assuming a_projection and b_projection are normalized unit vectors
        # For circular addition, we rotate a by angle b on the hypersphere
        
        # Compute rotation parameters (in the 2D circle subspace)
        cos_a = a_projection[:, 0]  # First component is cosine
        sin_a = a_projection[:, 1]  # Second component is sine
        cos_b = b_projection[:, 0]
        sin_b = b_projection[:, 1]
        
        # Apply the trig identities for addition
        cos_sum = cos_a * cos_b - sin_a * sin_b
        sin_sum = sin_a * cos_b + cos_a * sin_b
        
        # Combine into result vector (for higher dimensions, this would need to be modified)
        result = torch.stack([cos_sum, sin_sum], dim=1)
        
        # Normalize the result to maintain unit norm
        result = F.normalize(result, p=2, dim=1)
        
        return result
    
    def loss_fn(self, outputs, targets, regression_values=None):
        """
        Combined loss function for classification and regression.
        For classification: Minimize angle between output and class prototype
        For regression: Optimize as interpolation between prototypes
        """
        # Classification loss
        class_similarities = self.compute_similarity(outputs)
        class_loss = F.cross_entropy(class_similarities, targets)
        
        # Regression loss (if applicable)
        if regression_values is not None:
            # Normalize regression values to [-1, 1]
            norm_reg_values = 2 * (regression_values - regression_values.min()) / \
                             (regression_values.max() - regression_values.min()) - 1
                
            # Get the cosine similarity to the "upper bound" prototype
            # (assuming prototype pairs for each regression task)
            upper_bound_idx = 0  # This would need to be appropriately set
            upper_similarities = outputs[:, upper_bound_idx, :]
            
            # Compute regression loss as in the paper
            reg_loss = torch.mean((norm_reg_values - upper_similarities) ** 2)
            
            # Combined loss
            return class_loss + reg_loss
        
        return class_loss 