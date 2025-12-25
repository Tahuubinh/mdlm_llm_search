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
    
    # Free memory after initial property calculation
    del best_sequences, best_prop_values, best_distances
    torch.cuda.empty_cache()
    
    # Get top-k token indices for each position from x_theta
    # x_theta_probs shape: (batch_size, seq_len, vocab_size)
    topk_values, topk_indices = torch.topk(x_theta_probs, k=top_k_values_for_local_search, dim=-1)
    # topk_indices shape: (batch_size, seq_len, top_k_values_for_local_search)
    
    # Free topk_values as we only need indices
    del topk_values
    
    print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens in parallel...")
    
    # FULLY VECTORIZED neighbor generation across ALL ranks at once!
    # Generate neighbors for ALL ranks simultaneously
    all_neighbor_tokens = []
    all_neighbor_metadata = []
    
    for k_rank in range(top_k_values_for_local_search):
        # Extract candidate tokens for this rank: [batch_size, seq_len]
        candidate_tokens = topk_indices[:, :, k_rank]  # Shape: [batch_size, seq_len]
        
        # Create a mask for positions where candidate != current token
        is_different = candidate_tokens != best_tokens  # Shape: [batch_size, seq_len]
        
        # Get indices of all positions that differ
        batch_indices, pos_indices = torch.where(is_different)
        
        if len(batch_indices) == 0:
            # No valid neighbors for this rank
            del candidate_tokens, is_different, batch_indices, pos_indices
            continue
        
        # Create neighbors for this rank
        neighbor_tokens_rank = best_tokens[batch_indices].clone()
        neighbor_candidate_tokens = candidate_tokens[batch_indices, pos_indices]
        neighbor_tokens_rank[torch.arange(len(batch_indices)), pos_indices] = neighbor_candidate_tokens
        
        # Store neighbors and metadata
        all_neighbor_tokens.append(neighbor_tokens_rank)
        
        # Create metadata: (batch_idx, pos, rank)
        for b_idx, p_idx in zip(batch_indices.cpu().tolist(), pos_indices.cpu().tolist()):
            all_neighbor_metadata.append((b_idx, p_idx, k_rank))
        
        # Free temporary tensors
        del candidate_tokens, is_different, batch_indices, pos_indices, neighbor_tokens_rank, neighbor_candidate_tokens
    
    # Free topk_indices after generating all neighbors
    del topk_indices
    
    if len(all_neighbor_tokens) == 0:
        print(f"  Local search: No valid neighbors found, returning original tokens")
        del all_neighbor_tokens, all_neighbor_metadata, best_distances_tensor
        return best_tokens
    
    # Concatenate all neighbors into one big batch
    neighbor_tokens_batch = torch.cat(all_neighbor_tokens, dim=0)  # [total_neighbors, seq_len]
    del all_neighbor_tokens
    
    total_neighbors = neighbor_tokens_batch.shape[0]
    print(f"  Local search: Generated {total_neighbors} unique neighbors across all ranks")
    
    # Decode all sequences at once (no chunking for decode)
    neighbor_sequences = tokenizer.batch_decode(neighbor_tokens_batch.cpu().numpy(), skip_special_tokens=True)
    
    # # Decode in chunks to save memory
    # chunk_size = 50  # Decode 50 sequences at a time
    # neighbor_sequences = []
    # for i in range(0, total_neighbors, chunk_size):
    #     chunk = neighbor_tokens_batch[i:i+chunk_size]
    #     chunk_decoded = tokenizer.batch_decode(chunk.cpu().numpy(), skip_special_tokens=True)
    #     neighbor_sequences.extend(chunk_decoded)
    #     del chunk, chunk_decoded

    print(f"  Local search: Evaluating all {total_neighbors} neighbors in parallel...")
    
    # Compute properties for ALL neighbors in ONE call (this is the key speedup!)
    neighbor_prop_values = [
        calc(neighbor_sequences, total_neighbors, device)
        for calc in property_calcs_parallel
    ]
    
    # Free neighbor_sequences after property calculation
    del neighbor_sequences
    
    # Apply distance functions
    neighbor_distances = [
        dist_func(prop_value)
        for dist_func, prop_value in zip(distance_to_bounds_parallel, neighbor_prop_values)
    ]
    
    # Free neighbor_prop_values after distance calculation
    del neighbor_prop_values
    
    # Stack into [total_neighbors, num_properties]
    neighbor_distances_tensor = torch.stack(neighbor_distances, dim=-1)
    
    # Free neighbor_distances after stacking
    del neighbor_distances
    
    print(f"  Local search: Updating best sequences...")
    
    # Update best sequences if neighbors are better
    for i, (batch_idx, pos, k_rank) in enumerate(all_neighbor_metadata):
        neighbor_dist = neighbor_distances_tensor[i]  # [num_properties]
        best_dist = best_distances_tensor[batch_idx]  # [num_properties]
        
        # Convert to lists for compare_hierarchical
        neighbor_dist_list = neighbor_dist.cpu().tolist()
        best_dist_list = best_dist.cpu().tolist()
        
        # Check if neighbor is better
        if compare_hierarchical(neighbor_dist_list, best_dist_list) < 0:
            best_tokens[batch_idx] = neighbor_tokens_batch[i]
            # Update distances tensor with clone to avoid reference issues
            best_distances_tensor[batch_idx] = neighbor_dist.clone()
    
    # Clear memory after processing all neighbors
    del neighbor_tokens_batch, all_neighbor_metadata, neighbor_distances_tensor
    torch.cuda.empty_cache()
    
    # Free best_distances_tensor at the end
    del best_distances_tensor
    
    return best_tokens


if __name__ == "__main__":
    print("Language local search utilities for batch processing")
