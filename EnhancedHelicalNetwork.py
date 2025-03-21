import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from HelicalHypersphericalNetwork import HelicalHypersphericalNetwork

class EnhancedHelicalNetwork(HelicalHypersphericalNetwork):
    """
    Enhanced Helical Hyperspherical Network with additional capabilities:
    1. Learnable operation-specific subspaces
    2. Decoding mechanism to map back from helical representation to scalar values
    3. Support for multiple operations
    4. More flexible helical basis functions
    """
    
    def __init__(self, input_dim=1, output_dim=64, num_helices=9, 
                 helix_periods=[2, 3, 5, 7, 10, 20, 50, 100, 1000],
                 learnable_periods=False):
        super(EnhancedHelicalNetwork, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_helices=num_helices,
            helix_periods=helix_periods
        )
        
        # Whether periods should be learnable parameters
        self.learnable_periods = learnable_periods
        
        if learnable_periods:
            # Initialize learnable periods
            self.learnable_period_params = nn.Parameter(
                torch.tensor(helix_periods, dtype=torch.float32)
            )
        
        # Enhanced feature extraction network with residual connections
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        # Separate projections for different operations
        self.operations = ["addition", "subtraction", "multiplication", "squared", "complex"]
        self.operation_projections = nn.ModuleDict({
            op: nn.ModuleList([nn.Linear(128, output_dim) for _ in range(num_helices)])
            for op in self.operations
        })
        
        # Decoder network to map from helical representations back to scalar values
        self.decoder = nn.Sequential(
            nn.Linear(output_dim * num_helices, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Operation-specific weights to combine helical representations
        self.operation_weights = nn.ParameterDict({
            op: nn.Parameter(torch.ones(num_helices) / num_helices)
            for op in self.operations
        })
        
    def get_periods(self):
        """Get the current periods, either fixed or learnable."""
        if self.learnable_periods:
            # Ensure periods are always positive
            return F.softplus(self.learnable_period_params)
        else:
            return torch.tensor(self.helix_periods, device=self.prototypes.device)
    
    def _project_to_helix_subspace(self, features, period_idx, operation="addition"):
        """
        Project features to a specific helix subspace for a specific operation.
        
        Args:
            features: Batch of feature vectors [batch_size, feature_dim]
            period_idx: Index of the period to use for projection
            operation: Which operation projection to use
            
        Returns:
            Projection of features onto the helix subspace
        """
        if operation not in self.operations:
            raise ValueError(f"Unknown operation: {operation}. Must be one of {self.operations}")
            
        projection = self.operation_projections[operation][period_idx](features)
        return F.normalize(projection, p=2, dim=1)
    
    def forward(self, x, operation="addition"):
        """
        Forward pass with operation-specific projection.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            operation: Which operation to use for the projection
            
        Returns:
            Helical representations for each period
        """
        # Extract features
        features = self.feature_net(x)
        
        # Project to multiple helical manifolds with operation-specific projections
        helix_outputs = []
        for i, period in enumerate(self.helix_periods):
            projection = self._project_to_helix_subspace(features, i, operation)
            helix_outputs.append(projection)
        
        # Combine helical representations
        combined = torch.stack(helix_outputs, dim=1)
        
        return combined
    
    def compute_operation(self, a_projection, b_projection, period_idx, operation="addition"):
        """
        Compute the result of an operation using helical representations.
        
        Args:
            a_projection: Projection of 'a' value onto helix subspace [batch_size, output_dim]
            b_projection: Projection of 'b' value onto helix subspace [batch_size, output_dim]
            period_idx: Index of the period to use
            operation: Which operation to compute
            
        Returns:
            Predicted representation for the operation result
        """
        if operation == "addition":
            # Use the existing Clock algorithm
            return self.compute_clock_algorithm(a_projection, b_projection, period_idx)
        
        elif operation == "subtraction":
            # For subtraction, we invert the second value and add
            # Inverting in the circle means flipping the sign of the sine component
            inverted_b = torch.cat([
                b_projection[:, 0:1],                  # cosine stays the same
                -b_projection[:, 1:2]                  # sine gets inverted
            ], dim=1)
            
            # Then use addition algorithm
            return self.compute_clock_algorithm(a_projection, inverted_b, period_idx)
        
        elif operation == "multiplication":
            # For multiplication, we use the fact that multiplication in the complex plane
            # corresponds to adding angles and multiplying magnitudes
            # We'll approximate this with the cosine/sine representation
            
            # Extract components
            cos_a, sin_a = a_projection[:, 0], a_projection[:, 1]
            cos_b, sin_b = b_projection[:, 0], b_projection[:, 1]
            
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            cos_result = cos_a * cos_b - sin_a * sin_b
            sin_result = sin_a * cos_b + cos_a * sin_b
            
            # Combine into result
            result = torch.stack([cos_result, sin_result], dim=1)
            
            # Normalize to maintain unit norm
            return F.normalize(result, p=2, dim=1)
        
        elif operation == "squared":
            # Squaring means multiplying by itself
            return self.compute_operation(a_projection, a_projection, period_idx, "multiplication")
        
        elif operation == "complex":
            # Complex operation (a+b)² = a² + 2ab + b²
            a_squared = self.compute_operation(a_projection, a_projection, period_idx, "multiplication")
            b_squared = self.compute_operation(b_projection, b_projection, period_idx, "multiplication")
            ab = self.compute_operation(a_projection, b_projection, period_idx, "multiplication")
            
            # Double ab
            two_ab = torch.cat([
                ab[:, 0:1] * 2,  # Double the cosine
                ab[:, 1:2] * 2   # Double the sine
            ], dim=1)
            
            # Add the components: a² + 2ab
            temp = self.compute_operation(a_squared, two_ab, period_idx, "addition")
            
            # Add b²: (a² + 2ab) + b²
            return self.compute_operation(temp, b_squared, period_idx, "addition")
        
        else:
            raise ValueError(f"Unknown operation: {operation}. Must be one of {self.operations}")
    
    def encode(self, x, operation="addition"):
        """Encode a number into its helical representation."""
        return self(x, operation)
    
    def decode(self, representation):
        """
        Decode a helical representation back to a scalar value.
        
        Args:
            representation: Helical representation [batch_size, num_helices, output_dim]
            
        Returns:
            Decoded scalar value [batch_size, 1]
        """
        batch_size = representation.shape[0]
        
        # Flatten the representation
        flat_repr = representation.view(batch_size, -1)
        
        # Pass through decoder network
        return self.decoder(flat_repr)
    
    def compute_weighted_operation(self, a, b, operation="addition"):
        """
        Compute an operation with weighted contributions from all periods.
        
        Args:
            a: First input tensor [batch_size, input_dim]
            b: Second input tensor [batch_size, input_dim]
            operation: Which operation to compute
            
        Returns:
            Predicted scalar result [batch_size, 1]
        """
        # Get operation-specific weights
        weights = F.softmax(self.operation_weights[operation], dim=0)
        
        # Encode inputs
        a_reps = self.encode(a, operation)
        b_reps = self.encode(b, operation)
        
        # Compute operation for each period
        results = []
        for i in range(len(self.helix_periods)):
            a_period = a_reps[:, i, :2]  # Use first 2 dimensions (cos/sin)
            b_period = b_reps[:, i, :2]
            
            # Apply operation
            result_repr = self.compute_operation(a_period, b_period, i, operation)
            results.append(result_repr)
        
        # Weighted combination
        weighted_results = torch.zeros_like(results[0])
        for i, result in enumerate(results):
            weighted_results += weights[i] * result
        
        # Normalize
        weighted_results = F.normalize(weighted_results, p=2, dim=1)
        
        # Return the combined results
        return weighted_results
    
    def forward_with_decode(self, a, b, operation="addition"):
        """
        End-to-end forward pass with decoding to scalar value.
        
        Args:
            a: First input tensor [batch_size, input_dim]
            b: Second input tensor [batch_size, input_dim]
            operation: Which operation to compute
            
        Returns:
            Predicted scalar result [batch_size, 1]
        """
        # Encode inputs
        a_reps = self.encode(a, operation)
        b_reps = self.encode(b, operation)
        
        # Compute weighted operation across all periods
        batch_size = a.shape[0]
        combined_repr = torch.zeros(batch_size, self.num_helices, self.output_dim, device=a.device)
        
        for i in range(self.num_helices):
            a_period = a_reps[:, i, :]
            b_period = b_reps[:, i, :]
            
            # For simplicity, we only use the first 2 dimensions for the operation
            a_period_cossin = a_period[:, :2]
            b_period_cossin = b_period[:, :2]
            
            # Apply operation
            result_cossin = self.compute_operation(a_period_cossin, b_period_cossin, i, operation)
            
            # Fill in the result
            combined_repr[:, i, :2] = result_cossin
            # The rest of the dimensions remain 0
        
        # Decode to scalar value
        return self.decode(combined_repr) 