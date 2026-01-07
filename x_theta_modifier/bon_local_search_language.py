import torch
from x_theta_modifier.local_search_language_utils import local_search_language_batch
from .base import XThetaModifier
from .shared_ops import *

class BoNLocalSearchLanguageXThetaModifier(XThetaModifier):
    """
    BoN with Local Search specifically optimized for language data.
    Uses batch processing and model's x_theta probabilities to guide search.
    """
    
    def __init__(self, top_k_values_for_local_search=10, local_search_sampling_method='top_p', locally_typical_alpha=0.0, best_sequence_rank=1, *args, **kwargs):
        """Initialize the BoN Local Search sampler for language.
        
        Args:
            top_k_values_for_local_search: Number of top-k tokens to try per position
            local_search_sampling_method: Sampling method ('top_p', 'locally_typical', or 'locally_typical_distance')
            locally_typical_alpha: Bias parameter (interpretation depends on method):
                                   - For 'locally_typical': additive bias (0.0 = pure, >0 = bias toward high prob)
                                   - For 'locally_typical_distance': entropy scaling (1.0 = pure, <1.0 = bias toward high prob)
            best_sequence_rank: Select the sequence with the Nth smallest distance (1=best, 2=second best, etc.)
        """
        super().__init__(*args, **kwargs)
        self.top_k_values_for_local_search = top_k_values_for_local_search
        self.local_search_sampling_method = local_search_sampling_method
        self.locally_typical_alpha = locally_typical_alpha
        self.best_sequence_rank = best_sequence_rank
        
        if self.local_search_sampling_method == 'locally_typical':
            print(f"Initialized BoN Local Search for Language with top_k={self.top_k_values_for_local_search}, sampling_method={self.local_search_sampling_method}, alpha={self.locally_typical_alpha} (additive), best_sequence_rank={self.best_sequence_rank}")
        elif self.local_search_sampling_method == 'locally_typical_distance':
            print(f"Initialized BoN Local Search for Language with top_k={self.top_k_values_for_local_search}, sampling_method={self.local_search_sampling_method}, alpha={self.locally_typical_alpha} (scaling), best_sequence_rank={self.best_sequence_rank}")
        else:
            print(f"Initialized BoN Local Search for Language with top_k={self.top_k_values_for_local_search}, sampling_method={self.local_search_sampling_method}, best_sequence_rank={self.best_sequence_rank}")

    def get_x_theta_method(self):
        modify_x_theta = modify_x_theta_no_condition
        print("Using language local search (no validity condition).")
        
        def _language_local_search(x_theta, xt, step, best_clean_samples):
            # all_samples = gumbel_sample(x_theta, self.num_x_theta_samples)
            all_samples = gumbel_sample_sequential(x_theta, self.num_x_theta_samples)
            # Replace values in all_samples with corresponding values from xt
            # at positions where non_mask is True, keeping other values unchanged.
            all_samples, non_mask = keep_nonmask_values(all_samples, xt, self.mask_index)
            num_x_theta_samples_keepbest = self.num_x_theta_samples
            if best_clean_samples is not None:  # Check if best_clean_samples is not empty
                best_clean_samples_expanded = best_clean_samples.unsqueeze(0)
                all_samples = torch.cat([all_samples, best_clean_samples_expanded], dim=0)
                num_x_theta_samples_keepbest += 1

            device = all_samples.device
            batch_shape = x_theta.shape[:-1]
            batch_size, seq_len = batch_shape
            tokenizer = self.tokenizer

            # step_to_begin_local_search = 100
            # Only compute properties (BoN + Local Search) for late steps
            if step % 1 != 0:
                # Early steps: just take first sample without property computation
                best_tokens = all_samples[0]  # Shape: [batch_size, seq_len]
            else:
                # Late steps: perform BoN selection + local search
                # Optimization: Skip BoN selection if only 1 sample (no need to compare)
                if num_x_theta_samples_keepbest <= 1:
                    # Just take the first (and only) sample without computing properties
                    best_tokens = all_samples[0]  # Shape: [batch_size, seq_len]
                    old_best_tokens = best_tokens.clone()
                else:
                    # Normal BoN: compute properties and select best
                    print(f"Step {step}: BoN selection from {num_x_theta_samples_keepbest} samples")
                    best_tokens = find_best_tokens(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, self.property_calcs_parallel, self.distance_to_bounds_parallel)
            
                    old_best_tokens = best_tokens.clone()

                # Apply language-specific batch local search
                print(f"Step {step}: Starting batch local search for {batch_size} sequences...")
                x_theta_probs = torch.softmax(x_theta, dim=-1)  # Convert logits to probabilities
                # CRITICAL: Clear GPU cache before local search to free up memory from diffusion model
                # torch.cuda.empty_cache()
                # torch.cuda.synchronize()
                
                best_tokens = local_search_language_batch(
                    best_tokens=best_tokens,
                    x_theta_probs=x_theta_probs,
                    distance_to_bounds_parallel=self.distance_to_bounds_parallel,
                    property_calcs_parallel=self.property_calcs_parallel,
                    tokenizer=tokenizer,
                    top_k_values_for_local_search=self.top_k_values_for_local_search,
                    sampling_method=self.local_search_sampling_method,
                    locally_typical_alpha=self.locally_typical_alpha,
                    best_sequence_rank=self.best_sequence_rank,
                    filter_repetitions=self.filter_repetitions,
                    device=device
                )
                print(f"Step {step}: Local search completed")

            # # Apply language-specific batch local search
            # print(f"Step {step}: Starting batch local search for {batch_size} sequences...")
            # x_theta_probs = torch.softmax(x_theta, dim=-1)  # Convert logits to probabilities
            # best_tokens = local_search_language_batch(
            #     best_tokens=best_tokens,
            #     x_theta_probs=x_theta_probs,
            #     distance_to_bounds_parallel=self.distance_to_bounds_parallel,
            #     property_calcs_parallel=self.property_calcs_parallel,
            #     tokenizer=tokenizer,
            #     top_k_values_for_local_search=self.top_k_values_for_local_search,
            #     device=device
            # )
            # print(f"Step {step}: Local search completed")

            new_x_theta, validity_mask = modify_x_theta(best_tokens, tokenizer, x_theta, batch_size, device)
            if validity_mask is not None and validity_mask.any():
                new_best_tokens = torch.where(validity_mask.bool(), best_tokens, old_best_tokens)
                return new_x_theta, new_best_tokens
            return new_x_theta, best_tokens
        
        return _language_local_search
