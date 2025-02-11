# Language Models Use Trigonometry to Do Addition

This repository contains the implementation code for reproducing the experiments from the paper "Language Models Use Trigonometry to Do Addition" by Subhash Kantamneni and Max Tegmark.

## Overview

This project investigates how language models (specifically GPT-J, Pythia-6.9B, and Llama3.1-8B) represent and manipulate numbers to perform addition. The key findings show that these models:

1. Represent numbers as generalized helices
2. Use the "Clock" algorithm to perform addition
3. Implement this through specific attention heads and MLPs

## Setup

### Prerequisites

- Python 3.10 or later
- Conda package manager
- Git
- (Optional) NVIDIA GPU with CUDA support

### Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Activate the conda environment:
```bash
conda activate helix
```

## Project Structure

```
project/
├── configs/              # Configuration files
├── src/                  # Source code
│   ├── data/            # Dataset handling
│   ├── models/          # Model loading and validation
│   ├── analysis/        # Analysis tools
│   └── visualization/   # Plotting utilities
├── tests/               # Test files
├── notebooks/           # Jupyter notebooks
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Running Experiments

The main experiments from the paper can be reproduced using the notebooks in the `notebooks/` directory:

1. `1_data_exploration.ipynb`: Analyze number representations
2. `2_helix_analysis.ipynb`: Investigate helical structure

## Hardware Requirements

- Minimum 16GB RAM
- For GPU acceleration: NVIDIA GPU with at least 8GB VRAM
- Storage: ~10GB free space

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kantamneni2024language,
  title={Language Models Use Trigonometry to Do Addition},
  author={Kantamneni, Subhash and Tegmark, Max},
  journal={arXiv preprint arXiv:2502.00873},
  year={2024}
}
```

## License

[Add license information]

## Contributing

[Add contribution guidelines] 