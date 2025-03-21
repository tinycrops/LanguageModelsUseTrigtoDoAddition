# Helical Hyperspherical Network Documentation

## Mathematical Foundation

### Helical Representations

In the context of this project, a helical representation is a way to encode numbers using a combination of linear and circular components. The key insight is that language models (LLMs) naturally represent numbers as points on helices with specific periods.

A helix with period T maps a number n to a point in a higher-dimensional space where:
- The position along the helix corresponds to the linear component (n itself)
- The position around the helix (the phase) corresponds to n mod T

Mathematically, a helix function for a number n with period T can be represented as:
```
f_T(n) = [cos(2πn/T), sin(2πn/T), n/T]
```

Where:
- cos(2πn/T) and sin(2πn/T) represent the circular component
- n/T represents the linear component

### The Clock Algorithm for Addition

The Clock algorithm is a mathematical technique for performing addition by using the helical representation of numbers. It works by rotating one number's representation by the amount specified by the other number.

For two numbers a and b, their sum can be computed by:
1. Representing a and b as points on a helix with period T
2. Rotating the position of a by an amount proportional to b

This rotation can be performed using the trigonometric addition identities:
```
cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
```

By applying these identities to the circular components of the helical representation, we effectively perform addition in the modular space of the helix's period.

## Implementation Details

### HelicalHypersphericalNetwork

The `HelicalHypersphericalNetwork` class is a PyTorch neural network that explicitly models helical representations for numbers and implements the Clock algorithm for addition.

Key components:

1. **Feature Extraction Network**: Transforms input numbers into a high-dimensional feature representation.

2. **Helix Projections**: Projects features onto multiple helical manifolds with different periods.

3. **Hyperspherical Embeddings**: Ensures representations lie on a unit hypersphere to maintain consistent normalization.

4. **Clock Algorithm Implementation**: Implements the trigonometric identities to perform addition on the helical representations.

### Code Structure

- `HelicalHypersphericalNetwork.py`: Main model implementation
- `helical_network_demo.py`: Demonstration script with visualizations
- `train_helical_network.py`: Training script for addition tasks

## Usage Guide

### Training the Network

```python
from HelicalHypersphericalNetwork import HelicalHypersphericalNetwork
import torch

# Initialize model
model = HelicalHypersphericalNetwork(
    input_dim=1,               # Dimension of input (single number)
    output_dim=32,             # Dimension of helical embeddings
    num_helices=4,             # Number of helical representations
    helix_periods=[2, 5, 10, 100]  # Periods for each helix
)

# Train model (see train_helical_network.py for complete example)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Forward pass
a_reps = model(a_tensor)  # Shape: [batch_size, num_helices, output_dim]
b_reps = model(b_tensor)  # Shape: [batch_size, num_helices, output_dim]

# Compute predicted sum using Clock algorithm
for i in range(len(model.helix_periods)):
    # Extract cosine/sine components
    a_cossin = a_reps[:, i, :2]
    b_cossin = b_reps[:, i, :2]
    
    # Apply Clock algorithm
    sum_pred = model.compute_clock_algorithm(a_cossin, b_cossin, i)
    
    # Calculate loss
    loss = criterion(sum_pred, expected_sum_cossin)
```

### Visualizing Helical Representations

The demo script includes visualization functions for:

1. **Helical Representations**: Plots the first two PCA components of the helical representations for different periods, showing how numbers are organized in circular patterns.

2. **Clock Algorithm Demonstration**: Visualizes the addition process by showing the input numbers, their representations, and the predicted sum for different periods.

## Extending the Implementation

### Adding More Periods

To experiment with different period settings:

```python
model = HelicalHypersphericalNetwork(
    input_dim=1,
    output_dim=32,
    num_helices=6,  # Increased number of helices
    helix_periods=[2, 3, 5, 7, 10, 100]  # Added prime numbers
)
```

### Enhancing the Feature Extraction Network

For more complex number relationships, you can enhance the feature extraction network:

```python
class EnhancedHelicalNetwork(HelicalHypersphericalNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace feature network with a more powerful one
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
```

### Supporting Multiple Operations

To extend beyond addition to other operations like multiplication:

```python
def compute_multiplication(self, a_projection, b_projection, period_idx):
    """Implement multiplication using helical representations"""
    # Implementation would depend on specific operation
    # For multiplication in log space (multiplication = addition of logs)
    # You would need to transform the inputs and outputs accordingly
    pass
```

## Performance Considerations

- **Period Selection**: The most important periods identified in the paper are 2, 5, 10, and 100, which have specific numerical significance. Adjusting these can impact how well the model learns.

- **Embedding Dimensionality**: Higher-dimensional embeddings can capture more nuanced relationships but require more computation and may be prone to overfitting.

- **Training Data**: The quality and quantity of addition examples significantly impact model performance. Consider generating diverse examples that cover the full range of your target number space.

## References

- "Language Models Use Trigonometry to Do Addition" - Original paper describing helical representations in language models
- Hyperspherical embeddings literature from machine learning research 