"""Posterior sampling utilities for discrete diffusion models."""
from properties.peptides import *
from properties.molecules import *
from properties.trna import *
from properties.toxicity_property import calculate_toxicity, calc_toxicity_parallel
from properties.perplexity_property import calculate_perplexity, calc_perplexity_parallel
from properties.repetition_property import count_repetition_violations, calc_repetition_violations_parallel
from properties.quality_property import count_quality_violations, calc_quality_violations_parallel
from properties.property_util import *


class Sampler:
    """Base class for samplers with common initialization and abstract sampling method."""

    def __init__(self, num_posterior_samples, num_mcts_samples, property_type, lower_bound, upper_bound, none_distances, num_local_searches, tokenizer, top_k_categorical, nucleus, num_diffusion_steps, data, diffusion_model, vocab_size):
        """Initialize the sampler with common parameters.
        
        Args:
            num_posterior_samples: Number of posterior samples to generate
            num_mcts_samples: Number of MCTS samples to generate
            property_type: Type of property to consider during sampling
            lower_bound: Lower bound for the property
            upper_bound: Upper bound for the property
            none_distances: Distance value when property is None
            num_local_searches: Number of local searches
            tokenizer: Tokenizer for decoding sequences
            top_k_categorical: Top-k filtering parameter
            nucleus: Nucleus sampling parameter
            num_diffusion_steps: Total number of diffusion steps
            data: Dataset configuration
            diffusion_model: The diffusion model for MCTS rollouts
            vocab_size: Vocabulary size
        """
        self.vocab_size = vocab_size
        self.num_posterior_samples = num_posterior_samples
        self.num_mcts_samples = num_mcts_samples
        self.property_type = property_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.none_distances = none_distances
        self.none_distances_parallel = none_distances
        self.num_local_searches = num_local_searches
        self.tokenizer = tokenizer
        self.special_ids = set(self.tokenizer.special_tokens_map.values())
        self.top_k_categorical = top_k_categorical
        self.nucleus = nucleus
        self.num_diffusion_steps = num_diffusion_steps
        self.data = data
        self.diffusion_model = diffusion_model
        # Get mask_index directly from diffusion_model to ensure consistency
        self.mask_index = diffusion_model.mask_index

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
                "quality": calc_quality_violations_parallel,
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
                "quality": count_quality_violations,
            }.get(prop)
            if calc_func is None:
                raise ValueError(f"Unsupported property type: {prop}")
            self.property_calcs.append(calc_func)
        

        self.num_properties = len(self.property_type)
        if none_distances is None:
            self.none_distances_parallel = [float('inf')] * self.num_properties
            self.none_distances = [math.inf] * self.num_properties

        self.distance_to_bounds_parallel = [
            make_distance_to_bounds_parallel(lower_bound=lb, upper_bound=ub, none_distance=nd) 
            for lb, ub, nd in zip(self.lower_bound, self.upper_bound, self.none_distances_parallel)
        ]

        self.distance_to_bounds = [
            make_distance_to_bounds(lower_bound=lb, upper_bound=ub, none_distance=nd) 
            for lb, ub, nd in zip(self.lower_bound, self.upper_bound, self.none_distances)
        ]

    def get_sampling_method(self):
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Child classes must implement the get_sampling_method function.")
