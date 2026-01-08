
from properties.peptides import *
from properties.molecules import *
from properties.trna import *
from properties.toxicity_property import calculate_toxicity, calc_toxicity_parallel
from properties.perplexity_property import calculate_perplexity, calc_perplexity_parallel
from properties.repetition_property import count_repetition_violations, calc_repetition_violations_parallel
from properties.property_util import *


class XThetaModifier:
    """Base class for samplers with common initialization and abstract sampling method."""

    def __init__(self, num_x_theta_samples, property_type, lower_bound, upper_bound, num_local_searches, tokenizer, top_k_categorical, num_diffusion_steps, data, max_candidate_tokens, vocab_size, mask_index=None, top_k_values_for_local_search=10, local_search_sampling_method='top_p', locally_typical_alpha=0.0):
        """Initialize the sampler with common parameters.
        
        Args:
            num_x_theta_samples: Number of posterior samples to generate
            property_type: Type of property to consider during sampling
            lower_bound: Lower bound for the property
            upper_bound: Upper bound for the property
            vocab_size: Vocabulary size
            mask_index: Index of the mask token (if None, will be computed)
            top_k_values_for_local_search: Top-k tokens to try per position (language only)
            local_search_sampling_method: Sampling method for local search (language only)
            locally_typical_alpha: Bias parameter (interpretation depends on method, language only)
        """
        self.vocab_size = vocab_size
        self.num_x_theta_samples = num_x_theta_samples
        self.property_type = property_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.atom_candidates = atom_candidates
        self.branch_candidates = branch_candidates
        self.ring_candidates = ring_candidates
        self.num_local_searches = num_local_searches
        self.tokenizer = tokenizer
        self.special_ids = set(self.tokenizer.special_tokens_map.values())
        self.top_k_categorical = top_k_categorical
        self.num_diffusion_steps = num_diffusion_steps
        self.data = data
        self.max_candidate_tokens = max_candidate_tokens
        # Use provided mask_index if available, otherwise compute it
        if mask_index is not None:
            self.mask_index = mask_index
        elif (not hasattr(self.tokenizer, 'mask_token')
            or tokenizer.mask_token is None):
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id

        # Create a list of property calculation functions based on property_type
        self.property_calcs_parallel = []
        for prop in property_type:
            calc_func = {
                "qed": calc_qed_parallel,
                "sa": calc_sa_parallel,
                "valid": calc_validity_parallel,
                "peptide_length": calc_peptide_length_parallel,
                "hydrophobic_ratio": calc_hydrophobic_ratio_parallel,
                "net_charge": calc_net_charge_parallel,
                "trna_length": calc_trna_length_parallel,
                "acceptor_stem": count_acceptor_stem_parallel,
                "t_loop": get_t_loop_score_parallel,
                "fold": check_fold_structure_constraints_parallel,
                "toxicity": calc_toxicity_parallel,
                "perplexity": calc_perplexity_parallel,
                "repetition": calc_repetition_violations_parallel,
            }.get(prop)
            if calc_func is None:
                raise ValueError(f"Unsupported property type: {prop}")
            self.property_calcs_parallel.append(calc_func)

        self.property_calcs = []
        for prop in property_type:
            calc_func = {
                "qed": calc_qed,
                "sa": calc_sa,
                "valid": calc_validity,
                "peptide_length": calculate_peptide_length,
                "hydrophobic_ratio": calculate_hydrophobic_ratio,
                "net_charge": calculate_net_charge,
                "trna_length": calc_trna_length,
                "acceptor_stem": count_acceptor_stem,
                "t_loop": get_t_loop_score,
                "fold": check_fold_structure_constraints,
                "toxicity": calculate_toxicity,
                "perplexity": calculate_perplexity,
                "repetition": count_repetition_violations,
            }.get(prop)
            if calc_func is None:
                raise ValueError(f"Unsupported property type: {prop}")
            self.property_calcs.append(calc_func)

        self.distance_to_bounds_parallel = [
            make_distance_to_bounds_parallel(lower_bound=lb, upper_bound=ub) 
            for lb, ub in zip(self.lower_bound, self.upper_bound)
        ]

        self.distance_to_bounds = [
            make_distance_to_bounds(lower_bound=lb, upper_bound=ub) 
            for lb, ub in zip(self.lower_bound, self.upper_bound)
        ]

    def get_x_theta_method(self):
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Child classes must implement the get_x_theta_method function.")