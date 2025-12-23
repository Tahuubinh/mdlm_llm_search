import torch
from .base import Sampler
from .shared_ops import *
from properties.property_util import *

class MonteCarloTreeSearchSampler(Sampler):
    """Monte Carlo Tree Search sampler for sampling from categorical distributions."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the Monte Carlo Tree Search sampler."""
        super().__init__(*args, **kwargs)

    def _rollout_from_state(self, x_current, step):
        """Perform MCTS rollout from current state to generate clean sequences.
        
        Uses standard diffusion sampling (no modifications) for rollout.
        
        Args:
            x_current: Current state tensor of shape (B, L) where B is batch size
            step: Current diffusion step (0 to num_diffusion_steps-1, where 0 is most noisy)
            
        Returns:
            clean_sequences: Tensor of shape (B, num_mcts_samples, L) containing clean sequences
        """
        if self.diffusion_model is None:
            raise ValueError("Diffusion model has not been set. Call set_diffusion_model() first.")
        
        B, L = x_current.shape
        # Calculate remaining steps from NEXT step to end
        # step is current position (0 to num_diffusion_steps-1)
        # We rollout from step+1 to num_diffusion_steps-1
        remaining_steps = self.num_diffusion_steps - step - 1
        
        # If already at final step, just return current state repeated
        if remaining_steps <= 0:
            return x_current.unsqueeze(1).repeat(1, self.num_mcts_samples, 1)
        
        # For MCTS rollout, generate num_mcts_samples different trajectories
        # This simulates the inner loop of _diffusion_sample, independent from main loop
        all_rollouts = []
        # Calculate timestep (matching the main sampling loop)
        eps = 1e-5
        timesteps = torch.linspace(1, eps, self.num_diffusion_steps + 1, device=x_current.device)
        dt = (1 - eps) / self.num_diffusion_steps
        
        # for _ in range(self.num_mcts_samples):
        #     # Start from current state
        #     xt = x_current.clone()
            
        #     # Continue diffusion for remaining steps (standard sampling, no modifications)
        #     # Loop from step+1 to num_diffusion_steps-1
        #     for rollout_step in range(step + 1, self.num_diffusion_steps):
        #         with torch.no_grad():
        #             t = timesteps[rollout_step]
                    
        #             if hasattr(self.diffusion_model, 'T') and self.diffusion_model.T > 0:
        #                 t = (t * self.diffusion_model.T).to(torch.int)
        #                 t = t / self.diffusion_model.T
        #                 t += (1 / self.diffusion_model.T)
                    
        #             t_tensor = t * torch.ones(B, 1, device=xt.device)
                    
        #             # Get noise levels
        #             sigma_t, _ = self.diffusion_model.noise(t_tensor)
        #             sigma_s, _ = self.diffusion_model.noise(t_tensor - dt)
                    
        #             if sigma_t.ndim > 1:
        #                 sigma_t = sigma_t.squeeze(-1)
        #             if sigma_s.ndim > 1:
        #                 sigma_s = sigma_s.squeeze(-1)
                    
        #             move_chance_t = 1 - torch.exp(-sigma_t)
        #             move_chance_s = 1 - torch.exp(-sigma_s)
        #             move_chance_t = move_chance_t[:, None, None]
        #             move_chance_s = move_chance_s[:, None, None]
                    
        #             # Forward pass to get x_theta (NO modification)
        #             log_x_theta = self.diffusion_model(xt, sigma_t)
        #             x_theta = log_x_theta.exp()
                    
        #             # Compute posterior (NO modification)
        #             if self.diffusion_model.diffusion == 'absorbing_state':
        #                 q_xs = x_theta * (move_chance_t - move_chance_s)
        #                 q_xs[:, :, self.diffusion_model.mask_index] = move_chance_s[:, :, 0]
        #                 q_xs /= move_chance_t
        #             elif self.diffusion_model.diffusion == 'uniform':
        #                 q_xs = self.diffusion_model._compute_posterior(
        #                     x=x_theta,
        #                     xt=xt,
        #                     alpha_s=1 - move_chance_s,
        #                     alpha_t=1 - move_chance_t
        #                 )
        #             else:
        #                 raise NotImplementedError(
        #                     f"Diffusion type {self.diffusion_model.diffusion} not implemented.")
                    
        #             # Standard Gumbel sampling (NO posterior sampling modification)
        #             gumbel_norm = (1e-10 - (torch.rand_like(q_xs) + 1e-10).log())
        #             xs = (q_xs / gumbel_norm).argmax(dim=-1)
                    
        #             # Handle absorbing state mask
        #             if self.diffusion_model.diffusion == 'absorbing_state':
        #                 copy_flag = (xt != self.diffusion_model.mask_index).to(torch.bool)
        #                 xs = torch.where(copy_flag, xt, xs)
                    
        #             # Update for next iteration
        #             xt = xs
            
        #     all_rollouts.append(xt)

        for _ in range(self.num_mcts_samples):
            # Start from current state - each rollout is independent
            xt_rollout = x_current.clone()
            
            # Setup timesteps for this rollout (same as main loop)
            eps = 1e-5
            timesteps = torch.linspace(1, eps, self.num_diffusion_steps + 1, device=xt_rollout.device)
            dt = (1 - eps) / self.num_diffusion_steps
            
            # Continue diffusion for remaining steps (standard sampling, no modifications)
            # Loop from step+1 to num_diffusion_steps-1, matching main loop's for i in pbar
            for i in range(step + 1, self.num_diffusion_steps):
                with torch.no_grad():
                    # This matches exactly: for i in pbar: t = timesteps[i]
                    t = timesteps[i]
                    
                    if hasattr(self.diffusion_model, 'T') and self.diffusion_model.T > 0:
                        t = (t * self.diffusion_model.T).to(torch.int)
                        t = t / self.diffusion_model.T
                        t += (1 / self.diffusion_model.T)
                    
                    t_tensor = t * torch.ones(B, 1, device=xt_rollout.device)
                    
                    # Get noise levels
                    sigma_t, _ = self.diffusion_model.noise(t_tensor)
                    sigma_s, _ = self.diffusion_model.noise(t_tensor - dt)
                    
                    if sigma_t.ndim > 1:
                        sigma_t = sigma_t.squeeze(-1)
                    if sigma_s.ndim > 1:
                        sigma_s = sigma_s.squeeze(-1)
                    
                    move_chance_t = 1 - torch.exp(-sigma_t)
                    move_chance_s = 1 - torch.exp(-sigma_s)
                    move_chance_t = move_chance_t[:, None, None]
                    move_chance_s = move_chance_s[:, None, None]
                    
                    # Forward pass to get x_theta (NO modification)
                    log_x_theta = self.diffusion_model(xt_rollout, sigma_t)
                    x_theta = log_x_theta.exp()
                    
                    # Compute posterior (NO modification)
                    if self.diffusion_model.diffusion == 'absorbing_state':
                        q_xs = x_theta * (move_chance_t - move_chance_s)
                        q_xs[:, :, self.diffusion_model.mask_index] = move_chance_s[:, :, 0]
                        q_xs /= move_chance_t
                    elif self.diffusion_model.diffusion == 'uniform':
                        q_xs = self.diffusion_model._compute_posterior(
                            x=x_theta,
                            xt=xt_rollout,
                            alpha_s=1 - move_chance_s,
                            alpha_t=1 - move_chance_t
                        )
                    else:
                        raise NotImplementedError(
                            f"Diffusion type {self.diffusion_model.diffusion} not implemented.")
                    
                    # Apply nucleus sampling if enabled (same as main loop)
                    if self.nucleus:
                        q_xs = nucleus_sampling_filter(q_xs, self.nucleus)
                    
                    # Apply top-k filtering if enabled (same as main loop)
                    if self.top_k_categorical:
                        q_xs = filter_top_k_with_mask(q_xs, xt_rollout, self.mask_index, self.top_k_categorical)
                    
                    # Standard Gumbel sampling (NO posterior sampling modification)
                    gumbel_norm = (1e-10 - (torch.rand_like(q_xs) + 1e-10).log())
                    xs = (q_xs / gumbel_norm).argmax(dim=-1)
                    
                    # Handle absorbing state mask
                    if self.diffusion_model.diffusion == 'absorbing_state':
                        copy_flag = (xt_rollout != self.diffusion_model.mask_index).to(torch.bool)
                        # xs = torch.where(copy_flag, xt_rollout, xs)
                        q_xs[copy_flag] = 0.0
                        q_xs[copy_flag, xt_rollout[copy_flag]] = 1.0
                        xs = torch.where(copy_flag, xt_rollout, xs)

                    # Update for next iteration
                    xt_rollout = xs
            
            all_rollouts.append(xt_rollout)
        
        # Stack all rollouts: shape (num_mcts_samples, B, L) -> transpose to (B, num_mcts_samples, L)
        clean_sequences = torch.stack(all_rollouts, dim=0).transpose(0, 1)
        return clean_sequences

    def _evaluate_samples(self, all_samples, step):
        """Evaluate all samples using MCTS and select the best one.
        
        Args:
            all_samples: Tensor of shape (num_posterior_samples, B, L) containing candidate samples
            step: Current diffusion step
            
        Returns:
            best_sample: Tensor of shape (B, L) - the best sample selected
        """
        num_posterior_samples, B, L = all_samples.shape
        
        # Store expected distances for each sample
        # Structure: all_expected_distances[b][i] = [dist1, dist2, ..., distN] 
        # where b is batch index, i is sample index, N is number of properties
        all_expected_distances = [[None for _ in range(num_posterior_samples)] for _ in range(B)]
        
        for i in range(num_posterior_samples):
            # Get clean sequences from MCTS rollout for this sample
            # Shape: (B, num_mcts_samples, L)
            clean_sequences = self._rollout_from_state(all_samples[i], step)
            
            # Decode sequences to strings for property calculation
            # Shape: (B * num_mcts_samples, L)
            # Use reshape instead of view to handle non-contiguous tensors
            sequences_flat = clean_sequences.reshape(B * self.num_mcts_samples, L)
            decoded_sequences = self.tokenizer.batch_decode(sequences_flat, skip_special_tokens=True)
            
            # Process each batch element
            for b in range(B):
                # Get num_mcts_samples clean sequences for this batch element
                batch_sequences = decoded_sequences[b * self.num_mcts_samples : (b + 1) * self.num_mcts_samples]
                
                # For each property, calculate distances for all rollout samples
                # Structure: property_distances_all_rollouts[prop_idx] = [dist1, dist2, ..., dist_num_mcts_samples]
                property_distances_all_rollouts = [[] for _ in range(self.num_properties)]
                
                for seq in batch_sequences:
                    # Calculate distance for each property for this clean sequence
                    for prop_idx in range(self.num_properties):
                        prop_value = self.property_calcs[prop_idx](seq)
                        distance = self.distance_to_bounds[prop_idx](prop_value)
                        property_distances_all_rollouts[prop_idx].append(distance)
                
                # Calculate expected distance (mean) for each property
                expected_distances = [
                    sum(distances) / len(distances) 
                    for distances in property_distances_all_rollouts
                ]

                # expected_distances = [
                #     max(distances) 
                #     for distances in property_distances_all_rollouts
                # ]
                
                # Store expected distances for this sample and batch element
                all_expected_distances[b][i] = expected_distances

        # Select best sample for each batch element using hierarchical comparison
        best_samples = []
        for b in range(B):
            best_idx = 0
            best_distances = all_expected_distances[b][0]
            
            for i in range(1, num_posterior_samples):
                current_distances = all_expected_distances[b][i]
                if compare_hierarchical(current_distances, best_distances) < 0:
                    best_idx = i
                    best_distances = current_distances
            
            best_samples.append(all_samples[best_idx, b])
        
        # Stack to create output tensor of shape (B, L)
        return torch.stack(best_samples, dim=0)

    def get_sampling_method(self):
        """Return the sampling method function."""
        def _sample_categorical(categorical_probs, xt, step):
            """Sample from categorical distribution using MCTS.
            
            Args:
                categorical_probs: Probability distribution tensor of shape (B, L, vocab_size)
                xt: Current state tensor of shape (B, L)
                step: Current diffusion step
                
            Returns:
                Sampled indices of shape (B, L)
            """
            # Apply nucleus sampling if enabled
            if self.nucleus:
                categorical_probs = nucleus_sampling_filter(categorical_probs, self.nucleus)

            # Apply top-k filtering if enabled
            if self.top_k_categorical:
                categorical_probs = filter_top_k_with_mask(categorical_probs, xt, self.mask_index, self.top_k_categorical)

            # Step 1: Generate num_posterior_samples candidate samples using batched Gumbel sampling
            # all_samples = []
            # for _ in range(self.num_posterior_samples):
            #     gumbel_norm = (
            #         1e-10
            #         - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
            #     sample = (categorical_probs / gumbel_norm).argmax(dim=-1)
            #     all_samples.append(sample)
            # all_samples = torch.stack(all_samples, dim=0)  # Shape: (num_posterior_samples, B, L)

            
            # Generate all Gumbel noise at once: Shape (num_posterior_samples, B, L, vocab_size)
            gumbel_noise = (
                1e-10
                - (torch.rand(self.num_posterior_samples, *categorical_probs.shape, 
                             device=categorical_probs.device) + 1e-10).log()
            ).to(categorical_probs.dtype)
            
            # Broadcast categorical_probs to match gumbel_noise shape and divide
            # categorical_probs: (B, L, vocab_size) -> unsqueeze(0) -> (1, B, L, vocab_size)
            # gumbel_noise: (num_posterior_samples, B, L, vocab_size)
            # Result: (num_posterior_samples, B, L, vocab_size)
            all_samples = (categorical_probs.unsqueeze(0) / gumbel_noise).argmax(dim=-1)
            # Shape: (num_posterior_samples, B, L)

            # Step 2 & 3: Evaluate samples using MCTS and select the best
            best_sample = self._evaluate_samples(all_samples, step)
            
            return best_sample

        return _sample_categorical