from x_theta_modifier.local_search_utils import *
from .base import XThetaModifier

class StandardXThetaModifier(XThetaModifier):
    """Gumbel sampler for sampling from categorical distributions."""
    
    def __init__(self, top_k_values_for_local_search=10, local_search_sampling_method='top_p', locally_typical_alpha=0.0, best_sequence_rank=1, *args, **kwargs):
        """Initialize the Gumbel sampler."""
        super().__init__(*args, **kwargs)

    def get_x_theta_method(self):
        def _no_modification(x_theta, xt, step, best_clean_samples, prefix_lengths=None):
            return x_theta, xt
        
        return _no_modification