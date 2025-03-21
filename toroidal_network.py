import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ToroidalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, torus_dims=2, period_pairs=[(2, 5), (10, 100)]):
        """
        Toroidal Manifold Network
        
        Args:
            input_dim: Dimension of input data
            output_dim: Dimension of output embeddings per torus component
            torus_dims: Number of circular components in the torus (T^n)
            period_pairs: List of period pairs for each torus component [(p1_1, p1_2), (p2_1, p2_2), ...] 
        """
        super(ToroidalNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.torus_dims = torus_dims
        self.period_pairs = period_pairs
        
        assert len(period_pairs) == torus_dims, "Number of period pairs must match torus dimensions"
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Separate projection heads for each torus dimension
        self.circle_projections = nn.ModuleList([
            nn.Linear(128, output_dim) for _ in range(torus_dims * 2)  # 2 projections per torus dimension
        ])
        
        # Initialize prototypes uniformly on the torus
        self.prototypes = self._initialize_prototypes()
        
    def _initialize_prototypes(self, num_classes=10):
        """Initialize prototypes uniformly on the torus."""
        # For each class, create a prototype on the torus
        all_prototypes = []
        
        # Calculate angles for uniform distribution on each circle
        angles = torch.linspace(0, 2*math.pi, num_classes+1)[:-1]
        
        # For each class, generate a position on the torus
        for i in range(num_classes):
            # For a torus with n dimensions, we need n angles
            class_angles = angles[i].repeat(self.torus_dims * 2)  # Two circles per torus dimension
            
            # Generate prototype vector directly with sin/cos values
            prototype = torch.zeros(self.torus_dims * 4)  # 4 values per torus dimension (sin/cos for each circle)
            for j in range(self.torus_dims * 2):
                angle = class_angles[j]
                cos_idx = j * 2
                sin_idx = cos_idx + 1
                prototype[cos_idx] = torch.cos(angle)
                prototype[sin_idx] = torch.sin(angle)
                
            all_prototypes.append(prototype)
            
        prototypes = torch.stack(all_prototypes)
        
        return nn.Parameter(prototypes, requires_grad=False)
    
    def _project_to_circle(self, features, idx, period):
        """Project features to a circle with given period."""
        projection = self.circle_projections[idx](features)
        
        # Apply periodic activation - map to points on a circle
        angle = 2 * math.pi * projection / period
        
        # Return both sine and cosine components for complete circle coordinates
        cos_component = torch.cos(angle)
        sin_component = torch.sin(angle)
        
        return cos_component, sin_component
    
    def forward(self, x):
        """Forward pass mapping inputs to the torus manifold."""
        # Extract features
        features = self.feature_net(x)
        
        # Project to each circle of the torus
        circle_coords = []
        proj_idx = 0
        
        for dim in range(self.torus_dims):
            period1, period2 = self.period_pairs[dim]
            
            # Project to first circle
            cos1, sin1 = self._project_to_circle(features, proj_idx, period1)
            proj_idx += 1
            
            # Project to second circle
            cos2, sin2 = self._project_to_circle(features, proj_idx, period2)
            proj_idx += 1
            
            # Add all coordinates to the output
            circle_coords.extend([cos1, sin1, cos2, sin2])
        
        # Stack all coordinates to form the torus embedding
        torus_embedding = torch.stack(circle_coords, dim=1)
        
        return torus_embedding
    
    def compute_toroidal_distance(self, embed1, embed2):
        """Compute geodesic distance on the torus between two embeddings."""
        # Ensure proper shape for comparison
        batch_size = embed1.size(0)
        
        # Reshape embeddings to [batch, num_circles, 2] where each pair is (cos, sin)
        embed1_reshaped = embed1.reshape(batch_size, -1, 2)
        embed2_reshaped = embed2.reshape(batch_size, -1, 2)
        
        # Handle different number of circles (can happen with prototypes)
        num_circles1 = embed1_reshaped.size(1)
        num_circles2 = embed2_reshaped.size(1)
        
        # If the shapes don't match, only use the common dimensions
        if num_circles1 != num_circles2:
            min_circles = min(num_circles1, num_circles2)
            embed1_reshaped = embed1_reshaped[:, :min_circles, :]
            embed2_reshaped = embed2_reshaped[:, :min_circles, :]
        
        # Convert cartesian coordinates to angles
        angles1 = torch.atan2(embed1_reshaped[:, :, 1], embed1_reshaped[:, :, 0])  # [batch, num_circles]
        angles2 = torch.atan2(embed2_reshaped[:, :, 1], embed2_reshaped[:, :, 0])  # [batch, num_circles]
        
        # Compute angular distance on each circle (shortest arc on circle)
        angle_diff = torch.abs(angles1 - angles2)
        circle_distances = torch.min(angle_diff, 2*math.pi - angle_diff)
        
        # Compute Euclidean distance in the ambient space of the torus
        # (this is an approximation of the true geodesic distance)
        torus_distance = torch.sqrt(torch.sum(circle_distances**2, dim=1))
        
        return torus_distance
    
    def compute_similarity(self, outputs, prototypes=None):
        """Compute similarity to prototypes based on toroidal distance."""
        if prototypes is None:
            prototypes = self.prototypes
            
        # Reshape outputs to match prototype format if needed
        batch_size = outputs.size(0)
        outputs_flat = outputs.reshape(batch_size, -1)
        
        # Computer pairwise distances from outputs to all prototypes
        similarities = []
        for i in range(prototypes.size(0)):
            # Broadcast prototype to match batch size
            proto = prototypes[i].unsqueeze(0).expand(batch_size, -1)
            
            # Compute distance (smaller distance = higher similarity)
            distance = self.compute_toroidal_distance(outputs_flat, proto)
            
            # Convert distance to similarity (using negative exponential)
            similarity = torch.exp(-distance)
            similarities.append(similarity)
            
        # Stack similarities into a tensor [batch, num_prototypes]
        similarities = torch.stack(similarities, dim=1)
        
        return similarities
    
    def classification_loss(self, outputs, targets):
        """Classification loss based on prototype similarity."""
        similarities = self.compute_similarity(outputs)
        
        # Convert similarities to logits
        logits = F.log_softmax(similarities, dim=1)
        
        # Compute cross entropy loss
        loss = F.nll_loss(logits, targets)
        
        return loss
    
    def regression_loss(self, outputs, values, component_idx=0):
        """
        Regression loss for a specific torus component.
        Maps the angular value of the specified torus component to the regression target.
        
        Args:
            outputs: Torus embeddings [batch_size, num_features, output_dim]
            values: Regression targets (normalized to [0, 2π])
            component_idx: Which torus component to use for regression
        """
        batch_size = outputs.size(0)
        
        # For a 3D tensor output, extract the correct circle features
        # Given the output shape [batch_size, num_features, output_dim]
        if len(outputs.shape) == 3:
            # We need to extract sin and cos from the proper feature indices
            # Assuming the first two features (0, 1) of first dimension are cos and sin
            feature_idx = component_idx * 2  # Each component has 2 features (cos, sin)
            
            if feature_idx >= outputs.size(1):
                feature_idx = 0  # Use first feature as fallback
                
            # Extract cos and sin values for all batch elements at this feature index
            cos_values = outputs[:, feature_idx, :]  # [batch_size, output_dim]
            sin_values = outputs[:, feature_idx + 1, :]  # [batch_size, output_dim]
            
            # Average across the output dimension to get a single angle per sample
            cos_values = torch.mean(cos_values, dim=1)  # [batch_size]
            sin_values = torch.mean(sin_values, dim=1)  # [batch_size]
        else:
            # Handle flat output tensor (this was the previous logic)
            cos_idx = component_idx * 4
            sin_idx = cos_idx + 1
            
            if cos_idx >= outputs.size(1) or sin_idx >= outputs.size(1):
                cos_idx = 0
                sin_idx = 1
                
            cos_values = outputs[:, cos_idx]
            sin_values = outputs[:, sin_idx]
        
        # Convert to angles [0, 2π]
        pred_angles = torch.atan2(sin_values, cos_values)
        pred_angles = torch.where(pred_angles < 0, pred_angles + 2*math.pi, pred_angles)
        
        # Make sure values are the correct shape
        if len(values.shape) > 1 and values.shape[1] > 1:
            # If values has multiple dimensions, reshape to match pred_angles
            values = values[:, 0]  # Take just the first dimension
        
        # If pred_angles and values have different batch sizes, we need to fix that
        if pred_angles.size(0) != values.size(0):
            # This could happen with the last batch being smaller
            min_size = min(pred_angles.size(0), values.size(0))
            pred_angles = pred_angles[:min_size]
            values = values[:min_size]
        
        # Compute circular distance
        angle_diff = torch.abs(pred_angles - values)
        circle_distances = torch.min(angle_diff, 2*math.pi - angle_diff)
        
        # Mean squared circular distance as loss
        loss = torch.mean(circle_distances**2)
        
        return loss
    
    def combined_loss(self, outputs, targets=None, regression_values=None, regression_component=0):
        """Combined loss for classification and/or regression."""
        loss = 0
        
        if targets is not None:
            # Add classification loss
            class_loss = self.classification_loss(outputs, targets)
            loss += class_loss
            
        if regression_values is not None:
            # Add regression loss
            reg_loss = self.regression_loss(outputs, regression_values, regression_component)
            loss += reg_loss
            
        return loss 