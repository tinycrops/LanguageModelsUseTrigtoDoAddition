# Language Models Use Trig to Do Addition

This project demonstrates how language models utilize helical representations for numbers and the Clock algorithm for addition.

## Project Overview

Large language models need to represent numbers and perform arithmetic, but how do they actually accomplish this? This repository implements and analyzes the helical representation hypothesis, showing how neural networks can leverage trigonometric functions to perform operations like addition.

## Core Implementation Files

- `src/helix_fitting.py`: Implements helical fitting algorithms for neural network representations
- `src/utils.py`: Utility functions for data processing and analysis
- `src/number_representation.py`: Extracting number representations from models
- `src/causal_interventions.py`: Causal analysis of how models perform arithmetic
- `src/clock_algorithm.py`: Implementation of the Clock algorithm for addition
- `src/main.py`: Main experimental pipeline
- `src/demo.py`: Demo script showcasing the key findings
- `HelicalHypersphericalNetwork.py`: PyTorch implementation of a model that uses helical representations
- `helical_network_demo.py`: Demonstration of the helical network for addition

## Advanced Extensions

- `enhanced_helical_network.py`: Enhanced model supporting multiple operations
- `train_helical_network.py`: Training script for the basic model
- `advanced_training.py`: Advanced training script for model comparison
- `enhanced_demo.py`: Demonstration of the enhanced model capabilities
- `model_analysis.md`: Detailed analysis of the helical network properties

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Demo

Run the basic demo to see how language models use helical representations:

```bash
cd src
python demo.py
```

This extracts representations from a neural network, fits a helix, and visualizes the results. All output is saved to the `results/demo/` directory.

### Helical Network Demo

Try the helical network implementation that explicitly uses trigonometric representations:

```bash
python helical_network_demo.py
```

This generates visualizations showing how the helical network represents numbers and performs addition.

### Enhanced Model Demo

Explore the enhanced model that supports multiple operations:

```bash
python enhanced_demo.py
```

Outputs visualizations for addition, subtraction, multiplication, squared operations, and complex operations.

## Advanced Training

To compare different model architectures on arithmetic tasks:

```bash
python advanced_training.py
```

This trains and evaluates three architectures (Helical Network, MLP, Transformer) on five operations.

## Model Analysis and Comparison

We conducted a comprehensive comparison of the helical network architecture against standard MLPs and transformers. Key findings include:

### Performance Highlights

- **Rapid Convergence**: The helical network converges extremely quickly (often within 1-3 epochs) for operations with natural geometric interpretations like addition and subtraction.
- **Architectural Efficiency**: While MLPs achieved better final metrics through brute-force learning, they required significantly longer training with unstable learning curves.
- **Operation Suitability**: The helical representation excels at operations with clear circular interpretations and shows decent performance on other operations through composition.

### Architecture Advantages

1. **Strong Inductive Bias**: By building trigonometric operations directly into the architecture, the helical network dramatically reduces the search space during learning.
2. **Interpretable Representations**: The network's internal representations have clear geometric meaning, unlike black-box models.
3. **Parameter Efficiency**: Achieves competitive performance with far fewer parameters than general architectures.

For detailed analysis, see [Model Analysis](model_analysis.md) and [Model Comparison Report](model_comparison_report.md).

## Key Results

- Number representations in neural networks naturally organize into helical structures
- Addition emerges through the composition of circular rotations at different frequencies
- The Clock algorithm provides a mechanistic explanation for how models perform addition
- Specialized architectures leveraging these insights can perform mathematical operations with exceptional efficiency

## License

MIT 