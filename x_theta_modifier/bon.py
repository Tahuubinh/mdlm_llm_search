import torch
from properties.peptides import *
from properties.molecules import *
from properties.property_util import *
from .base import XThetaModifier
from .shared_ops import *

class BoNXThetaModifier(XThetaModifier):
    """Gumbel sampler for sampling from categorical distributions."""
    
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

            best_tokens = find_best_tokens(all_samples, device, seq_len, tokenizer, batch_size, num_x_theta_samples_keepbest, self.property_calcs_parallel, self.distance_to_bounds_parallel, prefix_lengths)

            new_x_theta, validity_mask = modify_x_theta(best_tokens, tokenizer, x_theta, batch_size, device)

            return new_x_theta, best_tokens
        
        return _no_modification