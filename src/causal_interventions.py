"""
Causal interventions module for testing the helix representation hypothesis.

This module implements activation patching experiments to provide causal evidence
that language models use helical representations for addition.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Any
import os
import tqdm
import random

from utils import to_numpy, format_prompt, save_results, load_results
from helix_fitting import generate_helix_basis, project_to_helix, reconstruct_from_helix

def get_residual_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    token_idx: int = -1,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract the residual stream at a specific layer and token.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        layer: Layer to extract from
        token_idx: Token index (-1 for last token)
        device: Device to run the model on
        
    Returns:
        np.ndarray: Vector representation of the residual stream
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # If token_idx is -1, use the last token
    if token_idx == -1:
        token_idx = inputs.input_ids.shape[1] - 1
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer][0, token_idx].clone()
        
    return to_numpy(hidden_states)

def activation_patching(
    model: Any,
    tokenizer: Any,
    clean_prompt: str,
    corrupted_prompt: str,
    layer: int,
    token_idx: int = -1,
    representation: Optional[np.ndarray] = None,
    device: str = "cuda"
) -> float:
    """
    Perform activation patching at a specific layer and token.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        clean_prompt: Clean prompt (a + b)
        corrupted_prompt: Corrupted prompt (a' + b or a + b')
        layer: Layer to patch
        token_idx: Token index (-1 for last token)
        representation: Optional replacement representation (if None, use the clean prompt's)
        device: Device to run on
        
    Returns:
        float: Logit difference for the correct answer
    """
    # Process prompts to get token IDs and compute correct answer
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(device)
    
    # If token_idx is -1, use the last token
    if token_idx == -1:
        token_idx = clean_inputs.input_ids.shape[1] - 1
    
    # Extract the correct answer from the clean prompt
    if "+" in clean_prompt:
        a, b = map(int, clean_prompt.split("+")[0].strip(), clean_prompt.split("+")[1].split("=")[0].strip())
        correct_answer = a + b
    else:
        raise ValueError("Clean prompt should contain an addition problem")
    
    # Get the token ID for the correct answer
    answer_token_id = tokenizer(str(correct_answer), add_special_tokens=False).input_ids[0]
    
    # Get clean hidden state if no replacement provided
    if representation is None:
        with torch.no_grad():
            outputs = model(**clean_inputs, output_hidden_states=True)
            clean_state = outputs.hidden_states[layer][0, token_idx].clone()
    else:
        clean_state = torch.tensor(representation, device=device)
    
    # Helper function to run model with patched activations
    def run_with_patch(inputs, patch_activation=False):
        # Create hooks for patching
        hooks = []
        patch_results = {}
        
        def create_hook(layer_idx):
            def hook(module, input, output):
                # Get the current hidden states output
                hidden_states = output[0] if isinstance(output, tuple) else output
                
                # If this is the layer we want to patch and patching is enabled
                if layer_idx == layer and patch_activation:
                    # Create a copy to modify
                    patched_hidden = hidden_states.clone()
                    # Replace the token representation
                    patched_hidden[0, token_idx] = clean_state
                    
                    # Return the patched hidden states
                    if isinstance(output, tuple):
                        output_list = list(output)
                        output_list[0] = patched_hidden
                        return tuple(output_list)
                    else:
                        return patched_hidden
                
                # If no patching, return original output
                return output
            
            return hook
        
        # Register hooks for each layer
        for i, layer_module in enumerate(model.transformer.h if hasattr(model, 'transformer') else model.model.layers):
            hooks.append(layer_module.register_forward_hook(create_hook(i)))
        
        # Run the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get logit for the correct answer
            answer_logit = logits[0, -1, answer_token_id].item()
            patch_results['answer_logit'] = answer_logit
            
            # Get top predicted token and its probability
            probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
            top_prob, top_token = torch.max(probs, dim=-1)
            patch_results['top_token'] = top_token.item()
            patch_results['top_prob'] = top_prob.item()
            
            # Check if top token matches correct answer
            patch_results['is_correct'] = (top_token.item() == answer_token_id)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return patch_results
    
    # Run with corrupted prompt without patching
    corrupted_results = run_with_patch(corrupted_inputs, patch_activation=False)
    
    # Run with corrupted prompt with patching
    patched_results = run_with_patch(corrupted_inputs, patch_activation=True)
    
    # Calculate logit difference
    logit_diff = patched_results['answer_logit'] - corrupted_results['answer_logit']
    
    return logit_diff

def generate_clean_corrupted_pairs(
    model_name: str,
    n_pairs: int = 100,
    a_range: List[int] = None,
    b_range: List[int] = None,
    corruption_type: str = "a"
) -> List[Dict[str, Union[str, int]]]:
    """
    Generate pairs of clean and corrupted prompts for activation patching.
    
    Args:
        model_name: Name of the model for prompt formatting
        n_pairs: Number of pairs to generate
        a_range: Range of values for a (default [0, 99])
        b_range: Range of values for b (default [0, 99])
        corruption_type: Type of corruption ("a" or "b")
        
    Returns:
        list: List of dictionaries with clean/corrupted prompts and related information
    """
    if a_range is None:
        a_range = list(range(100))
    if b_range is None:
        b_range = list(range(100))
    
    pairs = []
    while len(pairs) < n_pairs:
        # Sample a and b
        a = random.choice(a_range)
        b = random.choice(b_range)
        
        # Make sure a + b is below 200 (common single token limit)
        if a + b >= 200:
            continue
        
        # Create corrupted values
        if corruption_type == "a":
            # Corrupt a
            a_corrupted = random.choice([x for x in a_range if x != a])
            
            # Format prompts
            clean_prompt = format_prompt(a, b, model_name)
            corrupted_prompt = format_prompt(a_corrupted, b, model_name)
            
            pairs.append({
                'clean_prompt': clean_prompt,
                'corrupted_prompt': corrupted_prompt,
                'a': a,
                'b': b,
                'a_corrupted': a_corrupted,
                'answer': a + b
            })
        else:
            # Corrupt b
            b_corrupted = random.choice([x for x in b_range if x != b])
            
            # Format prompts
            clean_prompt = format_prompt(a, b, model_name)
            corrupted_prompt = format_prompt(a, b_corrupted, model_name)
            
            pairs.append({
                'clean_prompt': clean_prompt,
                'corrupted_prompt': corrupted_prompt,
                'a': a,
                'b': b,
                'b_corrupted': b_corrupted,
                'answer': a + b
            })
    
    return pairs

def patch_with_helix(
    model: Any,
    tokenizer: Any,
    prompt_pairs: List[Dict[str, Union[str, int]]],
    helix_fit: Dict[str, Any],
    layer: int,
    token_idx: int = -1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Perform activation patching using the fitted helix representation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_pairs: List of clean/corrupted prompt pairs
        helix_fit: Dictionary with helix fitting results
        layer: Layer to patch
        token_idx: Token index (-1 for last token)
        device: Device to run on
        
    Returns:
        dict: Dictionary with patching results
    """
    results = {
        'layer_patch': [],  # patching with actual clean hidden state
        'helix_patch': [],  # patching with helix reconstruction
        'pca_patch': [],    # patching with PCA baseline
        'random_patch': [], # patching with random values
        'prompt_pairs': prompt_pairs
    }
    
    for pair in tqdm.tqdm(prompt_pairs, desc=f"Patching layer {layer}"):
        clean_prompt = pair['clean_prompt']
        corrupted_prompt = pair['corrupted_prompt']
        
        # Get the clean hidden state
        clean_state = get_residual_stream(
            model, tokenizer, clean_prompt, layer, token_idx, device)
        
        # Standard activation patching with clean state
        layer_ld = activation_patching(
            model, tokenizer, clean_prompt, corrupted_prompt, 
            layer, token_idx, clean_state, device)
        
        # Project clean state to helix and reconstruct
        helix_projection = project_to_helix(clean_state, helix_fit)
        helix_reconstruction = reconstruct_from_helix(helix_projection, helix_fit)
        helix_ld = activation_patching(
            model, tokenizer, clean_prompt, corrupted_prompt, 
            layer, token_idx, helix_reconstruction, device)
        
        # PCA baseline: project to PCA space and back
        pca_components = helix_fit['pca_components']
        pca_dim = pca_components.shape[0]
        pca_projection = clean_state @ pca_components.T
        pca_reconstruction = pca_projection @ pca_components
        pca_ld = activation_patching(
            model, tokenizer, clean_prompt, corrupted_prompt, 
            layer, token_idx, pca_reconstruction, device)
        
        # Random baseline: use random values with same norm
        random_vec = np.random.randn(*clean_state.shape)
        random_vec = random_vec / np.linalg.norm(random_vec) * np.linalg.norm(clean_state)
        random_ld = activation_patching(
            model, tokenizer, clean_prompt, corrupted_prompt, 
            layer, token_idx, random_vec, device)
        
        # Store results
        results['layer_patch'].append(layer_ld)
        results['helix_patch'].append(helix_ld)
        results['pca_patch'].append(pca_ld)
        results['random_patch'].append(random_ld)
    
    # Compute averages
    for key in ['layer_patch', 'helix_patch', 'pca_patch', 'random_patch']:
        results[f'avg_{key}'] = np.mean(results[key])
    
    return results

def patch_across_layers(
    model: Any,
    tokenizer: Any,
    prompt_pairs: List[Dict[str, Union[str, int]]],
    helix_fit: Dict[str, Any],
    layers: List[int],
    token_idx: int = -1,
    device: str = "cuda",
    result_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform activation patching across multiple layers.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_pairs: List of clean/corrupted prompt pairs
        helix_fit: Dictionary with helix fitting results
        layers: List of layers to patch
        token_idx: Token index (-1 for last token)
        device: Device to run on
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with patching results across layers
    """
    all_results = {
        'layers': layers,
        'avg_layer_patch': [],
        'avg_helix_patch': [],
        'avg_pca_patch': [],
        'avg_random_patch': [],
        'per_layer_results': {}
    }
    
    for layer in layers:
        # Patch at this layer
        layer_results = patch_with_helix(
            model, tokenizer, prompt_pairs, helix_fit, 
            layer, token_idx, device)
        
        # Store average results
        all_results['avg_layer_patch'].append(layer_results['avg_layer_patch'])
        all_results['avg_helix_patch'].append(layer_results['avg_helix_patch'])
        all_results['avg_pca_patch'].append(layer_results['avg_pca_patch'])
        all_results['avg_random_patch'].append(layer_results['avg_random_patch'])
        
        # Store detailed results for this layer
        all_results['per_layer_results'][layer] = layer_results
    
    if result_path:
        save_results(all_results, result_path)
    
    return all_results

def patch_with_different_periods(
    model: Any,
    tokenizer: Any,
    prompt_pairs: List[Dict[str, Union[str, int]]],
    representations: np.ndarray,
    numbers: List[int],
    layer: int,
    token_idx: int = -1,
    period_combinations: List[List[int]] = None,
    device: str = "cuda",
    result_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform activation patching with different period combinations.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_pairs: List of clean/corrupted prompt pairs
        representations: Matrix of hidden representations
        numbers: List of numbers corresponding to the representations
        layer: Layer to patch
        token_idx: Token index (-1 for last token)
        period_combinations: List of period combinations to test
        device: Device to run on
        result_path: Optional path to save results
        
    Returns:
        dict: Dictionary with patching results for different periods
    """
    from helix_fitting import fit_helix
    
    # Default period combinations if none provided
    if period_combinations is None:
        period_combinations = [
            [100],                # Just linear trend
            [10, 100],            # Base 10 structure
            [5, 10, 100],         # Add T=5
            [2, 5, 10, 100],      # Full set from paper
        ]
    
    results = {
        'period_combinations': period_combinations,
        'avg_layer_patch': [],
        'avg_helix_patch': [],
        'param_counts': [],
        'detailed_results': []
    }
    
    # Get baseline with full layer patch
    layer_patch_results = []
    for pair in tqdm.tqdm(prompt_pairs, desc=f"Baseline layer patching"):
        clean_prompt = pair['clean_prompt']
        corrupted_prompt = pair['corrupted_prompt']
        
        # Standard activation patching
        layer_ld = activation_patching(
            model, tokenizer, clean_prompt, corrupted_prompt, 
            layer, token_idx, None, device)
        
        layer_patch_results.append(layer_ld)
    
    avg_layer_patch = np.mean(layer_patch_results)
    results['avg_layer_patch'] = avg_layer_patch
    
    # Test each period combination
    for periods in period_combinations:
        # Fit helix with these periods
        helix_fit = fit_helix(representations, numbers, periods, pca_dim=100)
        
        # Count parameters: 2*len(periods) + 1
        param_count = 2*len(periods) + 1
        results['param_counts'].append(param_count)
        
        # Patch with this helix
        helix_patch_results = []
        for pair in tqdm.tqdm(prompt_pairs, desc=f"Patching with periods {periods}"):
            clean_prompt = pair['clean_prompt']
            corrupted_prompt = pair['corrupted_prompt']
            
            # Get the clean hidden state
            clean_state = get_residual_stream(
                model, tokenizer, clean_prompt, layer, token_idx, device)
            
            # Project clean state to helix and reconstruct
            helix_projection = project_to_helix(clean_state, helix_fit)
            helix_reconstruction = reconstruct_from_helix(helix_projection, helix_fit)
            
            # Patch with reconstruction
            helix_ld = activation_patching(
                model, tokenizer, clean_prompt, corrupted_prompt, 
                layer, token_idx, helix_reconstruction, device)
            
            helix_patch_results.append(helix_ld)
        
        avg_helix_patch = np.mean(helix_patch_results)
        results['avg_helix_patch'].append(avg_helix_patch)
        
        # Store detailed results
        results['detailed_results'].append({
            'periods': periods,
            'helix_fit': helix_fit,
            'patch_results': helix_patch_results,
            'avg_patch': avg_helix_patch,
            'param_count': param_count,
            'relative_performance': avg_helix_patch / avg_layer_patch
        })
    
    if result_path:
        save_results(results, result_path)
    
    return results

def plot_patching_results(
    patching_results: Dict[str, Any],
    plot_path: Optional[str] = None
):
    """
    Plot activation patching results across layers.
    
    Args:
        patching_results: Dictionary with patching results
        plot_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    layers = patching_results['layers']
    
    # Plot results
    plt.plot(layers, patching_results['avg_layer_patch'], 'o-', 
             label='Layer Patch (Full)', linewidth=2)
    plt.plot(layers, patching_results['avg_helix_patch'], 's-', 
             label='Helix Fit', linewidth=2)
    plt.plot(layers, patching_results['avg_pca_patch'], '^-', 
             label='PCA Baseline', linewidth=2)
    plt.plot(layers, patching_results['avg_random_patch'], 'x-', 
             label='Random Baseline', linewidth=2)
    
    plt.xlabel('Layer')
    plt.ylabel('Logit Difference')
    plt.title('Activation Patching Results Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_period_comparison(
    period_results: Dict[str, Any],
    plot_path: Optional[str] = None
):
    """
    Plot comparison of different period combinations.
    
    Args:
        period_results: Dictionary with period comparison results
        plot_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get data
    param_counts = period_results['param_counts']
    helix_patches = period_results['avg_helix_patch']
    layer_patch = period_results['avg_layer_patch']
    period_combinations = period_results['period_combinations']
    
    # Plot data points
    plt.plot(param_counts, helix_patches, 'o-', linewidth=2, markersize=8)
    
    # Add reference line for full layer patch
    plt.axhline(y=layer_patch, linestyle='--', color='r', 
                label=f'Layer Patch: {layer_patch:.2f}')
    
    # Annotate points with period combinations
    for i, (params, patch, periods) in enumerate(zip(
            param_counts, helix_patches, period_combinations)):
        plt.annotate(f"T={periods}", 
                     (params, patch),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Logit Difference')
    plt.title('Effect of Different Period Combinations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    print("This module provides functions to perform activation patching experiments.")
    print("Import and use these functions in your main script or notebook.") 