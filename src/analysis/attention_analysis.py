from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from nnsight import LanguageModel
from .helix_fitting import HelixFitter

class AttentionAnalyzer:
    def __init__(
        self,
        model: LanguageModel,
        n_heads: int = 16,
        n_layers: int = 28
    ):
        """Initialize attention analyzer.
        
        Args:
            model: The language model to analyze
            n_heads: Number of attention heads per layer
            n_layers: Number of layers in the model
        """
        self.model = model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
    def get_head_effects(
        self,
        prompt: str,
        corrupted_prompt: str,
        layer: int,
        head: int
    ) -> Tuple[float, float]:
        """Calculate total and direct effects for a specific attention head.
        
        Args:
            prompt: Clean prompt
            corrupted_prompt: Corrupted version of prompt
            layer: Layer index
            head: Head index
            
        Returns:
            Tuple of (total_effect, direct_effect)
        """
        # Get clean run
        with self.model.invoke(prompt) as invoker:
            clean_head = self.model.model.transformer.h[layer].attn.output[0][:, head].save()
            clean_logits = self.model.lm_head.output.save()
            
        # Get corrupted run
        with self.model.invoke(corrupted_prompt) as invoker:
            corr_head = self.model.model.transformer.h[layer].attn.output[0][:, head].save()
            corr_logits = self.model.lm_head.output.save()
            
        # Patch head output
        with self.model.invoke(corrupted_prompt) as invoker:
            self.model.model.transformer.h[layer].attn.output[0][:, head] = clean_head.value
            patched_logits = self.model.lm_head.output.save()
            
        # Calculate effects
        total_effect = (patched_logits.value - corr_logits.value)[0, -1].mean().item()
        
        # For direct effect, patch only the path to logits
        with self.model.invoke(corrupted_prompt) as invoker:
            def direct_patch(x, layer_idx):
                if layer_idx == layer:
                    return clean_head.value
                return x
            self.model.model.transformer.h[layer].attn.output[0][:, head] = direct_patch(
                corr_head.value, layer
            )
            direct_logits = self.model.lm_head.output.save()
            
        direct_effect = (direct_logits.value - corr_logits.value)[0, -1].mean().item()
        
        return total_effect, direct_effect
        
    def analyze_heads(
        self,
        prompt: str,
        corrupted_prompt: str,
        top_k: int = 20
    ) -> Dict[str, List[Dict]]:
        """Analyze all attention heads and categorize them.
        
        Args:
            prompt: Clean prompt
            corrupted_prompt: Corrupted version of prompt
            top_k: Number of top heads to analyze in detail
            
        Returns:
            Dictionary containing categorized head information
        """
        # Get effects for all heads
        head_effects = []
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                te, de = self.get_head_effects(prompt, corrupted_prompt, layer, head)
                head_effects.append({
                    'layer': layer,
                    'head': head,
                    'total_effect': te,
                    'direct_effect': de,
                    'indirect_effect': te - de
                })
                
        # Sort by total effect
        head_effects.sort(key=lambda x: abs(x['total_effect']), reverse=True)
        top_heads = head_effects[:top_k]
        
        # Categorize heads
        ab_heads = []  # Heads that move a,b to last token
        ab_plus_heads = []  # Heads that output a+b
        mixed_heads = []  # Heads that do both
        
        for head in top_heads:
            # Calculate confidence scores
            ie_ratio = 1 - (head['direct_effect'] / head['total_effect'])
            de_ratio = head['direct_effect'] / head['total_effect']
            
            # Get attention patterns
            with self.model.invoke(prompt) as invoker:
                attn_pattern = (
                    self.model.model.transformer.h[head['layer']]
                    .attn.output[0][:, head['head']].save()
                )
            
            # Analyze attention patterns
            attn = attn_pattern.value[0].cpu().numpy()
            attends_to_ab = np.mean(attn[:-1]) > 0.5  # Attends to a,b tokens
            attends_to_last = attn[-1] > 0.5  # Attends to last token
            
            if attends_to_ab and ie_ratio > 0.7:
                ab_heads.append(head)
            elif attends_to_last and de_ratio > 0.7:
                ab_plus_heads.append(head)
            else:
                mixed_heads.append(head)
                
        return {
            'a_b_heads': ab_heads,
            'a_plus_b_heads': ab_plus_heads,
            'mixed_heads': mixed_heads
        }
        
    def validate_head_categories(
        self,
        heads: Dict[str, List[Dict]],
        helix_fitter: HelixFitter,
        prompt: str,
        a: int,
        b: int
    ) -> Dict[str, float]:
        """Validate head categorization using helix fitting.
        
        Args:
            heads: Categorized heads from analyze_heads
            helix_fitter: Fitted HelixFitter instance
            prompt: Example prompt
            a: First number in addition
            b: Second number in addition
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {}
        
        # Get helix representations
        helix_a = helix_fitter.transform(np.array([a]))
        helix_b = helix_fitter.transform(np.array([b]))
        helix_sum = helix_fitter.transform(np.array([a + b]))
        
        # Validate a,b heads
        ab_accuracy = 0
        for head in heads['a_b_heads']:
            with self.model.invoke(prompt) as invoker:
                output = (
                    self.model.model.transformer.h[head['layer']]
                    .attn.output[0][:, head['head']].save()
                )
            head_output = output.value[0, -1].cpu().numpy()
            
            # Check if output is closer to a,b helix than a+b helix
            ab_dist = np.mean([
                np.linalg.norm(head_output - helix_a),
                np.linalg.norm(head_output - helix_b)
            ])
            sum_dist = np.linalg.norm(head_output - helix_sum)
            ab_accuracy += float(ab_dist < sum_dist)
            
        metrics['ab_head_accuracy'] = ab_accuracy / len(heads['a_b_heads'])
        
        # Validate a+b heads similarly
        sum_accuracy = 0
        for head in heads['a_plus_b_heads']:
            with self.model.invoke(prompt) as invoker:
                output = (
                    self.model.model.transformer.h[head['layer']]
                    .attn.output[0][:, head['head']].save()
                )
            head_output = output.value[0, -1].cpu().numpy()
            
            sum_dist = np.linalg.norm(head_output - helix_sum)
            ab_dist = np.mean([
                np.linalg.norm(head_output - helix_a),
                np.linalg.norm(head_output - helix_b)
            ])
            sum_accuracy += float(sum_dist < ab_dist)
            
        metrics['sum_head_accuracy'] = sum_accuracy / len(heads['a_plus_b_heads'])
        
        return metrics 