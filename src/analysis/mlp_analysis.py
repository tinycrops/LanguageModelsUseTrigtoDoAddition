from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from nnsight import LanguageModel
from .helix_fitting import HelixFitter

class MLPAnalyzer:
    def __init__(
        self,
        model: LanguageModel,
        n_layers: int = 28,
        n_neurons: int = 16384
    ):
        """Initialize MLP analyzer.
        
        Args:
            model: The language model to analyze
            n_layers: Number of layers in the model
            n_neurons: Number of neurons per MLP layer
        """
        self.model = model
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        
    def get_mlp_effects(
        self,
        prompt: str,
        corrupted_prompt: str,
        layer: int
    ) -> Tuple[float, float]:
        """Calculate total and direct effects for an MLP layer.
        
        Args:
            prompt: Clean prompt
            corrupted_prompt: Corrupted version of prompt
            layer: Layer index
            
        Returns:
            Tuple of (total_effect, direct_effect)
        """
        # Get clean run
        with self.model.invoke(prompt) as invoker:
            clean_mlp = self.model.model.transformer.h[layer].mlp.output[0].save()
            clean_logits = self.model.lm_head.output.save()
            
        # Get corrupted run
        with self.model.invoke(corrupted_prompt) as invoker:
            corr_mlp = self.model.model.transformer.h[layer].mlp.output[0].save()
            corr_logits = self.model.lm_head.output.save()
            
        # Patch MLP output
        with self.model.invoke(corrupted_prompt) as invoker:
            self.model.model.transformer.h[layer].mlp.output[0] = clean_mlp.value
            patched_logits = self.model.lm_head.output.save()
            
        # Calculate effects
        total_effect = (patched_logits.value - corr_logits.value)[0, -1].mean().item()
        
        # For direct effect, patch only the path to logits
        with self.model.invoke(corrupted_prompt) as invoker:
            def direct_patch(x, layer_idx):
                if layer_idx == layer:
                    return clean_mlp.value
                return x
            self.model.model.transformer.h[layer].mlp.output[0] = direct_patch(
                corr_mlp.value, layer
            )
            direct_logits = self.model.lm_head.output.save()
            
        direct_effect = (direct_logits.value - corr_logits.value)[0, -1].mean().item()
        
        return total_effect, direct_effect
        
    def analyze_neurons(
        self,
        prompt: str,
        layer: int,
        top_k: int = 100
    ) -> List[Dict]:
        """Analyze individual neurons in an MLP layer.
        
        Args:
            prompt: Input prompt
            layer: Layer to analyze
            top_k: Number of top neurons to analyze
            
        Returns:
            List of neuron information dictionaries
        """
        # Get neuron preactivations
        with self.model.invoke(prompt) as invoker:
            preacts = (
                self.model.model.transformer.h[layer].mlp.output[0].save()
            )
        
        # Calculate neuron importance scores
        preact_values = preacts.value[0, -1].cpu().numpy()
        importance = np.abs(preact_values)
        
        # Get top neurons
        top_indices = np.argsort(importance)[-top_k:]
        
        neurons = []
        for idx in top_indices:
            neurons.append({
                'index': idx,
                'layer': layer,
                'activation': preact_values[idx],
                'importance': importance[idx]
            })
            
        return neurons
        
    def fit_neuron_patterns(
        self,
        neurons: List[Dict],
        a: int,
        b: int,
        helix_fitter: HelixFitter
    ) -> List[Dict]:
        """Fit periodic patterns to neuron activations.
        
        Args:
            neurons: List of neuron information
            a: First number in addition
            b: Second number in addition
            helix_fitter: Fitted HelixFitter instance
            
        Returns:
            Updated neuron information with fitted patterns
        """
        # Get helix components
        helix_a = helix_fitter.transform(np.array([a]))
        helix_b = helix_fitter.transform(np.array([b]))
        helix_sum = helix_fitter.transform(np.array([a + b]))
        
        for neuron in neurons:
            # Project neuron activation onto helix components
            activation = neuron['activation']
            
            # Calculate correlations with helix components
            a_corr = np.corrcoef(activation, helix_a.flatten())[0, 1]
            b_corr = np.corrcoef(activation, helix_b.flatten())[0, 1]
            sum_corr = np.corrcoef(activation, helix_sum.flatten())[0, 1]
            
            # Determine primary function
            correlations = {
                'a': abs(a_corr),
                'b': abs(b_corr),
                'a+b': abs(sum_corr)
            }
            primary_function = max(correlations.items(), key=lambda x: x[1])[0]
            
            neuron.update({
                'a_correlation': a_corr,
                'b_correlation': b_corr,
                'sum_correlation': sum_corr,
                'primary_function': primary_function
            })
            
        return neurons
        
    def analyze_mlp_circuit(
        self,
        prompt: str,
        corrupted_prompt: str,
        min_layer: int = 14,
        max_layer: int = 27
    ) -> Dict[str, List[Dict]]:
        """Analyze the MLP circuit for addition.
        
        Args:
            prompt: Clean prompt
            corrupted_prompt: Corrupted version of prompt
            min_layer: First layer to analyze
            max_layer: Last layer to analyze
            
        Returns:
            Dictionary of categorized MLPs
        """
        # Analyze effects for each MLP
        mlp_effects = []
        for layer in range(min_layer, max_layer + 1):
            te, de = self.get_mlp_effects(prompt, corrupted_prompt, layer)
            mlp_effects.append({
                'layer': layer,
                'total_effect': te,
                'direct_effect': de,
                'indirect_effect': te - de
            })
            
        # Sort by total effect
        mlp_effects.sort(key=lambda x: abs(x['total_effect']), reverse=True)
        
        # Categorize MLPs
        builder_mlps = []  # MLPs that build a+b helix
        output_mlps = []   # MLPs that output to logits
        
        for mlp in mlp_effects:
            if mlp['layer'] <= 18:
                builder_mlps.append(mlp)
            else:
                output_mlps.append(mlp)
                
        return {
            'builder_mlps': builder_mlps,
            'output_mlps': output_mlps
        } 