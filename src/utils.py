"""
Utility functions for analyzing how LLMs use trigonometry to do addition.

Contains common functionality for loading models, extracting activations,
and data manipulation shared across the project.
"""

import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Union, Optional, Any

# Constants
DEFAULT_MODELS = {
    "gpt-j": "EleutherAI/gpt-j-6B",
    "pythia": "EleutherAI/pythia-6.9b",
    "llama": "meta-llama/Llama-3.1-8B",  # Note: Access to Llama may require authentication
}

PROMPTS = {
    "gpt-j": "Output ONLY a number. {a}+{b}=",
    "pythia": "Output ONLY a number. {a}+{b}=",
    "llama": "The following is a correct addition problem.\n{a}+{b}=",
}

# Helper functions
def load_model(model_name: str, device: str = "cuda") -> Tuple[Any, Any]:
    """
    Load a pre-trained language model and its tokenizer.
    
    Args:
        model_name: Name of the model (gpt-j, pythia, llama) or HuggingFace model path
        device: Device to load the model on (cuda or cpu)
        
    Returns:
        tuple: (model, tokenizer)
    """
    if model_name in DEFAULT_MODELS:
        model_path = DEFAULT_MODELS[model_name]
    else:
        model_path = model_name
        
    print(f"Loading model {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with appropriate precision
    if device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.to(device)
    
    return model, tokenizer

def generate_addition_dataset(a_range: List[int], b_range: List[int]) -> List[Dict[str, int]]:
    """
    Generate a dataset of addition problems.
    
    Args:
        a_range: Range of values for the first operand
        b_range: Range of values for the second operand
        
    Returns:
        list: List of dictionaries with keys 'a', 'b', and 'answer'
    """
    dataset = []
    for a in a_range:
        for b in b_range:
            dataset.append({
                'a': a,
                'b': b,
                'answer': a + b
            })
    return dataset

def format_prompt(a: int, b: int, model_name: str) -> str:
    """
    Format an addition prompt based on the model.
    
    Args:
        a: First operand
        b: Second operand
        model_name: Name of the model
        
    Returns:
        str: Formatted prompt
    """
    if model_name in PROMPTS:
        template = PROMPTS[model_name]
    else:
        template = "Output ONLY a number. {a}+{b}="
        
    return template.format(a=a, b=b)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to NumPy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        np.ndarray: NumPy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

def fftn_mean(activations: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the Fast Fourier Transform across a specified axis and average.
    
    Args:
        activations: Activation matrix of shape [num_items, dim]
        axis: Axis to compute FFT over (default 0)
        
    Returns:
        np.ndarray: Average magnitude of the Fourier components
    """
    # Center the activations
    centered = activations - np.mean(activations, axis=axis, keepdims=True)
    
    # Compute FFT
    fft_result = np.fft.fftn(centered, axes=[axis])
    
    # Calculate magnitude and average across dimensions
    magnitude = np.abs(fft_result)
    mean_magnitude = np.mean(magnitude, axis=tuple(range(1, magnitude.ndim)))
    
    return mean_magnitude

def save_results(results: Dict, filepath: str):
    """
    Save results to a file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.npy'):
        np.save(filepath, results)
    elif filepath.endswith('.npz'):
        np.savez(filepath, **results)
    else:
        np.save(f"{filepath}.npy", results)

def load_results(filepath: str) -> Dict:
    """
    Load results from a file.
    
    Args:
        filepath: Path to load the results from
        
    Returns:
        dict: Dictionary of results
    """
    if filepath.endswith('.npz'):
        return dict(np.load(filepath))
    else:
        return np.load(filepath, allow_pickle=True).item() 