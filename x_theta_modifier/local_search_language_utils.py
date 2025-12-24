"""
Language-specific local search utilities.
Optimized for batch processing with property models.
"""

import torch
from properties.property_util import compare_hierarchical


def local_search_language_batch(best_tokens, x_theta_probs, distance_to_bounds_parallel, property_calcs_parallel, tokenizer, top_k_values_for_local_search, device):
    """
    Batch local search for language data using top-k values from x_theta probabilities.
    For each position, only try the top-k most probable tokens.
    
    This function processes ALL sequences in the batch together for efficiency.
    Uses parallel property calculation functions for speed.
    
    Args:
        best_tokens: Current best sequences (batch_size x seq_len) tensor of token IDs
        x_theta_probs: Probability distribution over vocab (batch_size x seq_len x vocab_size)
        distance_to_bounds_parallel: List of distance functions (parallel version)
        property_calcs_parallel: List of property calculation functions (parallel version)
        tokenizer: Tokenizer for decoding sequences
        top_k_values_for_local_search: Number of top-k values to try per position
        device: Device to use
    
    Returns:
        best_tokens: Best sequences found (batch_size x seq_len) tensor
    """
    batch_size, seq_len = best_tokens.shape
    
    # Calculate initial properties for all sequences in batch (using parallel version)
    best_sequences = tokenizer.batch_decode(best_tokens.cpu().numpy(), skip_special_tokens=True)
    
    # Use parallel property calculation
    best_prop_values = [
        calc(best_sequences, batch_size, device)
        for calc in property_calcs_parallel
    ]
    # Apply distance functions to get distances (each is a tensor of shape [batch_size])
    best_distances = [
        dist_func(prop_value)
        for dist_func, prop_value in zip(distance_to_bounds_parallel, best_prop_values)
    ]
    # Stack into [batch_size, num_properties]
    best_distances_tensor = torch.stack(best_distances, dim=-1)
    
    # Get top-k token indices for each position from x_theta
    # x_theta_probs shape: (batch_size, seq_len, vocab_size)
    topk_values, topk_indices = torch.topk(x_theta_probs, k=top_k_values_for_local_search, dim=-1)
    # topk_indices shape: (batch_size, seq_len, top_k_values_for_local_search)
    
    # Iterate through each rank (1st best, 2nd best, ..., k-th best)
    for k_rank in range(top_k_values_for_local_search):
        print(f"  Local search: trying rank {k_rank+1}/{top_k_values_for_local_search} tokens...")
        
        # VECTORIZED neighbor generation (no Python loops!)
        # Extract candidate tokens for this rank: [batch_size, seq_len]
        candidate_tokens = topk_indices[:, :, k_rank]  # Shape: [batch_size, seq_len]
        
        # Create a mask for positions where candidate != current token
        # This helps us skip identical replacements
        is_different = candidate_tokens != best_tokens  # Shape: [batch_size, seq_len]
        
        # Count total neighbors we'll generate
        total_neighbors = is_different.sum().item()
        
        if total_neighbors == 0:
            # print(f"    No valid neighbors for rank {k_rank+1} (all candidates same as current), skipping...")
            continue
        
        # print(f"    Generating {total_neighbors} unique neighbors using vectorized operations...")
        
        # Method: For each (batch_idx, pos) pair where is_different[batch_idx, pos] == True,
        # create a neighbor by copying best_tokens[batch_idx] and changing position pos
        
        # Get indices of all positions that differ
        batch_indices, pos_indices = torch.where(is_different)
        # batch_indices: which batch element, pos_indices: which position
        
        # Create all neighbors at once using advanced indexing
        # Step 1: Repeat best_tokens for each neighbor we need to create
        # neighbor_tokens_batch = best_tokens[batch_indices].clone()  # [total_neighbors, seq_len]
        neighbor_tokens_batch = best_tokens[batch_indices].clone()
        
        # Step 2: Get the candidate token for each neighbor
        neighbor_candidate_tokens = candidate_tokens[batch_indices, pos_indices]  # [total_neighbors]
        
        # Step 3: Use scatter to place candidate tokens at the correct positions
        # We need to scatter neighbor_candidate_tokens into neighbor_tokens_batch at positions pos_indices
        neighbor_tokens_batch[torch.arange(total_neighbors), pos_indices] = neighbor_candidate_tokens
        
        # Create metadata for tracking which (batch_idx, pos) each neighbor corresponds to
        # We'll use this later for updating best_tokens
        all_neighbor_metadata = list(zip(batch_indices.cpu().tolist(), pos_indices.cpu().tolist()))
        
        # print(f"    Vectorized generation complete: {total_neighbors} neighbors created")
        
        # Evaluate ALL neighbors in ONE SINGLE batch using PARALLEL property calculation
        neighbor_sequences = tokenizer.batch_decode(neighbor_tokens_batch.cpu().numpy(), skip_special_tokens=True)
        
        # Compute properties for ALL neighbors in ONE call (this is the key speedup!)
        neighbor_prop_values = [
            calc(neighbor_sequences, total_neighbors, device)
            for calc in property_calcs_parallel
        ]
        # Apply distance functions
        neighbor_distances = [
            dist_func(prop_value)
            for dist_func, prop_value in zip(distance_to_bounds_parallel, neighbor_prop_values)
        ]
        # Stack into [total_neighbors, num_properties]
        neighbor_distances_tensor = torch.stack(neighbor_distances, dim=-1)
        
        # Update best sequences if neighbors are better
        for i, (batch_idx, pos) in enumerate(all_neighbor_metadata):
            neighbor_dist = neighbor_distances_tensor[i]  # [num_properties]
            best_dist = best_distances_tensor[batch_idx]  # [num_properties]
            
            # Convert to lists for compare_hierarchical
            neighbor_dist_list = neighbor_dist.cpu().tolist()
            best_dist_list = best_dist.cpu().tolist()
            
            # Check if neighbor is better
            if compare_hierarchical(neighbor_dist_list, best_dist_list) < 0:
                # Use neighbor_tokens_batch[i] instead of all_neighbor_tokens[i]
                best_tokens[batch_idx] = neighbor_tokens_batch[i]
                # Update distances tensor
                best_distances_tensor[batch_idx] = neighbor_dist
                # print(f"    Found better sequence for batch {batch_idx} at position {pos}")
    
    return best_tokens


if __name__ == "__main__":
    print("Language local search utilities for batch processing")
