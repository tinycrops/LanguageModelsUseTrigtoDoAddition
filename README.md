# Language Models Use Trigonometry to Do Addition

This repository contains an implementation of the "Clock Algorithm" described in the paper "Language Models Use Trigonometry to Do Addition" by Kantamneni and Tegmark (2025). The implementation includes both the original base-10 configuration and an adaptation to a base-20 number system.

## Overview

The Clock Algorithm explains how language models perform addition by representing numbers as generalized helices in a high-dimensional space and manipulating these helices using trigonometric identities. This implementation provides:

1. A Python implementation of the Clock Algorithm
2. Visualization tools for the helix representations
3. Support for both base-10 and base-20 number systems
4. Comparison and analysis tools for the two number systems

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## File Structure

- `ClockAlgorithm.py`: Main implementation of the Clock Algorithm
- `test_clock_algorithm.py`: Basic tests and visualizations for both base-10 and base-20
- `compare_bases.py`: Detailed comparison between base-10 and base-20 implementations
- `figures/`: Generated visualizations (created when running the scripts)

## The Clock Algorithm Explained

The Clock Algorithm works as follows:

1. **Helix Representation**: Numbers are represented as a combination of a linear component and multiple circular components with different periods.
   - For base-10: Periods T = [2, 5, 10, 100]
   - For base-20: Periods T = [2, 4, 5, 20, 400]
   
2. **Addition Mechanism**: To add two numbers, their helix representations are combined using trigonometric identities:
   - cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
   - sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
   
3. **Answer Decoding**: The combined helix is decoded to produce the final answer.

## Base-20 Adaptation

The base-20 adaptation modifies the periods used in the helix representation to account for the different number system:

- The base-10 modulus is 100 (largest period), representing numbers modulo 100
- The base-20 modulus is 400 (largest period), representing numbers modulo 400

The base-20 periods were chosen to include:
- T=2 for evenness
- T=4 as a factor of 20
- T=5 as a factor of 20
- T=20 as the base itself
- T=400 as the modulus (20²)

## How to Run

To run the basic test suite:

```bash
python test_clock_algorithm.py
```

To run the comparison between base-10 and base-20:

```bash
python compare_bases.py
```

## Visualizations

The scripts generate visualizations that show:

1. The helix representation of individual numbers
2. The addition process for specific number pairs
3. Comparison of accuracy between base-10 and base-20
4. Error patterns for both number systems
5. Analysis of how different periods contribute to addition accuracy

All visualizations are saved in the `figures/` directory.

## References

Kantamneni, S., & Tegmark, M. (2025). Language Models Use Trigonometry to Do Addition. arXiv:2502.00873v1. 