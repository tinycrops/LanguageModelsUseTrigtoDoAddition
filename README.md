# Toroidal Manifold Learning

This project explores toroidal manifold learning as an extension of helical hyperspherical networks for machine learning tasks. The implementation demonstrates how using toroidal geometries can improve representation efficiency for data with multiple periodic components.

## Overview

Geometric deep learning explores how different manifold structures affect neural network performance. This project specifically compares:

1. **Helical Hyperspherical Networks** - Embedding data onto hyperspheres with helical structure
2. **Toroidal Networks** - Embedding data onto torus manifolds (product of circles)

The hypothesis is that toroidal representations can better capture data with multiple independent periodic components.

## Files

- `toroidal_network.py` - Implementation of the Toroidal Network architecture
- `train_toroidal.py` - Training script for Toroidal Networks on synthetic data
- `experiment_comparison.py` - Comparison between Toroidal and Helical networks
- `paper.md` - Research paper outline explaining the theoretical foundation

## Requirements

```
torch
numpy
matplotlib
scikit-learn
```

## Usage

### Training a Toroidal Network

```bash
python train_toroidal.py --n_samples 2000 --epochs 100 --visualize
```

Optional arguments:
- `--n_samples`: Number of synthetic samples (default: 2000)
- `--noise_level`: Noise level for data generation (default: 0.1)
- `--output_dim`: Dimension of embeddings (default: 16)
- `--torus_dims`: Number of torus dimensions (default: 2)
- `--batch_size`: Training batch size (default: 64)
- `--learning_rate`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 50)
- `--task`: Task type: 'classification', 'regression', or 'both' (default: 'both')
- `--visualize`: Flag to visualize embeddings

### Comparing Networks

```bash
python experiment_comparison.py --n_samples 5000 --epochs 100
```

This script generates synthetic data with multiple periodic components and trains both Toroidal and Helical networks, comparing their performance on classification and regression tasks.

## Theory

Toroidal manifolds (T^n) are formed as the product of n circles. In contrast to hyperspherical embeddings, toroidal manifolds naturally represent multiple independent periodic variables. For example:

- Position on a single circle (S^1) can represent one periodic variable
- Position on a torus (S^1 × S^1) can represent two independent periodic variables

By projecting data onto toroidal manifolds, the network can better model cyclic relationships in multiple dimensions simultaneously, which is particularly useful for:

- Time series with multiple seasonal patterns
- Periodic physical processes
- Rotational/angular data with multiple components

## Results Visualization

After running the comparison experiment, two visualization files will be generated:
- `comparison_results.png` - Bar charts comparing accuracy, MSE, and training time
- `training_loss_curves.png` - Learning curves for both network types

## Future Work

- Extend to higher-dimensional toroidal manifolds
- Explore more complex manifold combinations
- Apply to real-world datasets with known periodic structure
- Investigate attention mechanisms for weighting different toroidal components 