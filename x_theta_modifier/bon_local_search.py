import torch
# Add RDKit SA_Score to path for sascorer import
from x_theta_modifier.local_search_utils import *
from .base import XThetaModifier
from .shared_ops import *
from properties.peptides import *
from properties.molecules import *
from properties.property_util import *

class BoNLocalSearchXThetaModifier(XThetaModifier):
    
    def __init__(self, top_k_values_for_local_search=10, *args, **kwargs):
        """Initialize the BoN Local Search sampler.
        
        Args:
            top_k_values_for_local_search: For language data, number of top-k tokens to try per position
        """
        super().__init__(*args, **kwargs)
        self.top_k_values_for_local_search = top_k_values_for_local_search

    def get_x_theta_method(self):
        if self.data == 'qm9':
            modify_x_theta = modify_x_theta_on_valid_best_tokens
            print("Using validity condition in XTheta Modifier.")
        else:
            modify_x_theta = modify_x_theta_no_condition
            print("Not using validity condition in XTheta Modifier.")

        # Determine local search function based on data type
        if self.data == 'qm9':
            local_search = local_search_qm9
            use_language_search = False
        elif self.data == 'grampa':
            local_search = local_search_grampa
            use_language_search = False
        elif self.data == 'trna':
            local_search = local_search_trna
            use_language_search = False
        elif self.data in ['openwebtext-split', 'openwebtext', 'lm1b', 'wikitext103', 'wikitext2']:
            # Language data - use new top-k local search
            local_search = None  # Not used for language
            use_language_search = True
            print(f"Using language local search with top_k={self.top_k_values_for_local_search}")
        else:
            raise ValueError(f"Local search not implemented for data type: {self.data}")
        
        def _no_modification(x_theta, xt, step, best_clean_samples, prefix_lengths=None):
            all_samples = gumbel_sample(x_theta, self.num_x_theta_samples)
            # Replace values in all_samples with corresponding values from xt
            # at positions where non_mask is True, keeping other values unchanged.
            all_samples, non_mask = keep_nonmask_values(all_samples, xt, self.mask_index)
            num_x_theta_samples_keepbest = self.num_x_theta_samples
            if best_clean_samples is not None:  # Check if best_clean_samples is not empty
                best_clean_samples_expanded = best_clean_samples.unsqueeze(0)  # Add a new dimension to match all_samples shape
                all_samples = torch.cat([all_samples, best_clean_samples_expanded], dim=0)  # Concatenate along the first dimension
                num_x_theta_samples_keepbest += 1

            device = all_samples.device
            batch_shape = x_theta.shape[:-1]
            batch_size, seq_len = batch_shape
            tokenizer = self.tokenizer

            # Optimization: Skip BoN selection if only 1 sample (no need to compare)
            if num_x_theta_samples_keepbest <= 1:
                # Just take the first (and only) sample without computing properties
                best_tokens = all_samples[0]  # Shape: [batch_size, seq_len]
                print("Skipping BoN selection (num_samples=1), using sample directly")
            else:
                # Normal BoN: compute properties and select best
                best_tokens = find_best_tokens(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, self.property_calcs_parallel, self.distance_to_bounds_parallel, prefix_lengths, self.property_type)
            
            old_best_tokens = best_tokens.clone()

            # Apply local search
            if use_language_search:
                # For language data, pass x_theta probabilities and top_k parameter
                x_theta_probs = torch.softmax(x_theta, dim=-1)  # Convert logits to probabilities
                best_tokens = local_search_on_best_tokens(
                    best_tokens, tokenizer, self.num_local_searches, 
                    self.distance_to_bounds, self.property_calcs, 
                    local_search, device, self.max_candidate_tokens,
                    x_theta=x_theta_probs, 
                    top_k_values_for_local_search=self.top_k_values_for_local_search
                )
            else:
                # For molecular/peptide data, use original local search
                best_tokens = local_search_on_best_tokens(
                    best_tokens, tokenizer, self.num_local_searches, 
                    self.distance_to_bounds, self.property_calcs, 
                    local_search, device, self.max_candidate_tokens
                )

            new_x_theta, validity_mask = modify_x_theta(best_tokens, tokenizer, x_theta, batch_size, device)
            if validity_mask is not None and validity_mask.any():
                new_best_tokens = torch.where(validity_mask.bool(), best_tokens, old_best_tokens)
                return new_x_theta, new_best_tokens
            return new_x_theta, best_tokens
        
        return _no_modification