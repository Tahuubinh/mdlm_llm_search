import torch
# Add RDKit SA_Score to path for sascorer import
from x_theta_modifier.local_search_utils import *
from .base import XThetaModifier
from .shared_ops import *
from properties.peptides import *
from properties.molecules import *
from properties.property_util import *

class BoNLocalSearchXThetaModifierLastStep(XThetaModifier):
    
    def __init__(self, *args, **kwargs):
        """Initialize the Gumbel sampler."""
        super().__init__(*args, **kwargs)

    def get_x_theta_method(self):
        if self.data == 'qm9':
            modify_x_theta = modify_x_theta_on_valid_best_tokens
            print("Using validity condition in XTheta Modifier.")
        else:
            modify_x_theta = modify_x_theta_no_condition
            print("Not using validity condition in XTheta Modifier.")

        if self.data == 'qm9':
            local_search = local_search_qm9
        elif self.data == 'grampa':
            local_search = local_search_grampa
        elif self.data == 'trna':
            local_search = local_search_trna
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
                best_tokens = find_best_tokens(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, self.property_calcs_parallel, self.distance_to_bounds_parallel)
            
            old_best_tokens = best_tokens.clone()

            if step == self.num_diffusion_steps - 1:
                best_tokens = local_search_on_best_tokens(best_tokens, tokenizer, self.num_local_searches, self.distance_to_bounds, self.property_calcs, local_search, device, self.max_candidate_tokens)

            new_x_theta, validity_mask = modify_x_theta(best_tokens, tokenizer, x_theta, batch_size, device)
            if validity_mask is not None and validity_mask.any():
                new_best_tokens = torch.where(validity_mask.bool(), best_tokens, old_best_tokens)
                return new_x_theta, new_best_tokens
            return new_x_theta, best_tokens
        
        return _no_modification