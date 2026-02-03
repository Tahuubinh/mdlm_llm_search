"""
Language-specific local search utilities.
Optimized for batch processing with property models.
"""

import torch
from properties.property_util import compare_hierarchical


def select_top_k_tokens(x_theta_probs, top_k, method='top_p', alpha=0.0):
    """
    Select top-k token indices from probability distribution.
    
    Args:
        x_theta_probs: Probability distribution (batch_size, seq_len, vocab_size)
        top_k: Number of tokens to select
        method: Selection method - 'top_p', 'locally_typical', or 'locally_typical_distance'
        alpha: Bias parameter (interpretation depends on method):
               - For 'locally_typical': additive bias weight (0.0 = pure, >0 = bias toward high prob)
               - For 'locally_typical_distance': entropy scaling factor (1.0 = pure, <1.0 = bias toward high prob)
    
    Returns:
        topk_indices: Tensor of shape (batch_size, seq_len, top_k) with selected token indices
    """
    if method == 'top_p':
        # Default: Select tokens with highest probabilities
        topk_values, topk_indices = torch.topk(x_theta_probs, k=top_k, dim=-1)
        return topk_indices
    
    elif method == 'locally_typical':
        # Locally Typical Sampling with probability bias (additive):
        # Formula: distance = |-log(p) - H| - alpha * log(p)
        # 
        # The alpha term biases selection toward high probability tokens:
        # - alpha = 0: pure locally typical (only entropy distance matters)
        # - alpha > 0: high prob tokens get smaller distance → more likely to be selected
        # - alpha → ∞: behaves like top-p (highest prob tokens always selected)
        
        # Calculate entropy H for each position: H = -sum(p * log(p))
        epsilon = 1e-10
        log_probs = torch.log(x_theta_probs + epsilon)
        entropy = -torch.sum(x_theta_probs * log_probs, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Calculate -log(p) for each token
        neg_log_probs = -log_probs  # (batch_size, seq_len, vocab_size)
        
        # Calculate base distance: |-log(p) - H|
        distance_to_entropy = torch.abs(neg_log_probs - entropy)  # (batch_size, seq_len, vocab_size)
        
        # Apply probability bias: subtract alpha * log(p)
        if alpha > 0:
            distance_to_entropy = distance_to_entropy - alpha * log_probs
        
        # Select top-k tokens with SMALLEST distance
        _, topk_indices = torch.topk(distance_to_entropy, k=top_k, dim=-1, largest=False)
        
        return topk_indices
    
    elif method == 'locally_typical_distance':
        # Locally Typical Sampling with entropy scaling (multiplicative):
        # Formula: distance = |-log(p) - alpha * H|
        #
        # The alpha parameter scales the target information (using alpha instead of tau for consistency):
        # - alpha = 1.0: pure locally typical (target = H, original algorithm)
        # - alpha < 1.0: scale entropy down → target moves toward 0 → prefer high prob tokens
        # - alpha → 0.0: target → 0 → equivalent to minimizing -log(p) → top-p behavior
        #
        # This is a different parameterization from 'locally_typical':
        # - 'locally_typical' uses additive bias: distance - alpha*log(p)
        # - 'locally_typical_distance' uses multiplicative scaling: |-log(p) - alpha*H|
        
        # Calculate entropy H for each position: H = -sum(p * log(p))
        epsilon = 1e-10
        log_probs = torch.log(x_theta_probs + epsilon)
        entropy = -torch.sum(x_theta_probs * log_probs, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Calculate -log(p) for each token
        neg_log_probs = -log_probs  # (batch_size, seq_len, vocab_size)
        
        # Calculate distance with scaled entropy: |-log(p) - alpha * H|
        # When alpha = 1.0: standard locally typical
        # When alpha → 0.0: target approaches 0, which means selecting high prob tokens (low -log(p))
        scaled_entropy = alpha * entropy
        distance_to_scaled_entropy = torch.abs(neg_log_probs - scaled_entropy)  # (batch_size, seq_len, vocab_size)
        
        # Select top-k tokens with SMALLEST distance
        _, topk_indices = torch.topk(distance_to_scaled_entropy, k=top_k, dim=-1, largest=False)
        
        return topk_indices
    
    else:
        raise ValueError(f"Unknown sampling method: {method}. Choose 'top_p', 'locally_typical', or 'locally_typical_distance'")


def clean_text_samples(text_samples):
    """
    Clean text samples by removing special tokens.
    This matches the cleaning logic in inference_utils.py.
    
    Args:
        text_samples: List of text strings
    
    Returns:
        List of cleaned text strings
    """
    cleaned = []
    for sample in text_samples:
        # Remove all special tokens (matching inference_utils.py logic)
        cleaned_text = sample.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<mask>', '').replace('<unk>', '').replace('<cls>', '').replace('<sep>', '').replace('<reserved>', '').strip()
        cleaned.append(cleaned_text)
    return cleaned


def remove_prefix_from_token_ids(token_ids, prefix_lengths):
    """
    Remove prefix tokens from token ID sequences before decoding.
    This ensures we decode only the generated portion, avoiding re-tokenization issues.
    
    Args:
        token_ids: Tensor of shape (batch_size, seq_len) with token IDs
        prefix_lengths: Tensor of shape (batch_size,) with prefix lengths
    
    Returns:
        Tensor of post-prefix token IDs with shape (batch_size, max_generated_len)
    """
    if prefix_lengths is None:
        return token_ids
    
    batch_size, seq_len = token_ids.shape
    post_prefix_token_ids = []
    
    for batch_idx, prefix_len in enumerate(prefix_lengths.cpu().tolist()):
        if prefix_len > 0 and prefix_len < seq_len:
            # Extract only post-prefix tokens
            post_prefix_tokens = token_ids[batch_idx, prefix_len:]
            post_prefix_token_ids.append(post_prefix_tokens)
        elif prefix_len >= seq_len:
            # Entire sequence is prefix (shouldn't happen, but handle it)
            # Return empty sequence
            post_prefix_token_ids.append(torch.tensor([], dtype=token_ids.dtype, device=token_ids.device))
        else:
            # No prefix, use full sequence
            post_prefix_token_ids.append(token_ids[batch_idx])
    
    # Pad to same length for batch processing
    if len(post_prefix_token_ids) > 0:
        max_len = max(len(t) for t in post_prefix_token_ids)
        # Use tokenizer's pad_token_id (typically 50256 for GPT-2)
        padded = torch.nn.utils.rnn.pad_sequence(post_prefix_token_ids, batch_first=True, padding_value=50256)
        return padded
    else:
        return token_ids


def local_search_language_batch(best_tokens, x_theta_probs, distance_to_bounds_parallel, property_calcs_parallel, tokenizer, top_k_values_for_local_search, sampling_method='top_p', locally_typical_alpha=0.0, best_sequence_rank=1, prefix_lengths=None, device='cuda', property_types=None):
    """
    Batch local search for language data using top-k values from x_theta probabilities.
    For each position, only try the top-k most probable tokens.
    
    This function processes ALL sequences in the batch together for efficiency.
    Uses parallel property calculation functions for speed.
    
    IMPORTANT: Follows discrete diffusion output processing:
    1. Decode token IDs → text (with special tokens)
    2. Clean text: remove special tokens
    3. Pass cleaned text to property calculators (they handle encoding internally)
    
    CRITICAL: Prefix Protection
    - If prefix_lengths is provided, prefix tokens will NEVER be modified
    - Only non-prefix positions are candidates for local search
    
    Args:
        best_tokens: Current best sequences (batch_size x seq_len) tensor of token IDs
        x_theta_probs: Probability distribution over vocab (batch_size x seq_len x vocab_size)
        distance_to_bounds_parallel: List of distance functions (parallel version)
        property_calcs_parallel: List of property calculation functions (parallel version)
        tokenizer: Tokenizer for decoding sequences
        top_k_values_for_local_search: Number of top-k values to try per position
        sampling_method: Selection method - 'top_p' (highest probability) or 'locally_typical' (closest to entropy)
        best_sequence_rank: Select the sequence with the Nth smallest distance (1=best, 2=second best, etc.)
        prefix_lengths: Optional tensor of shape (batch_size,) indicating prefix length for each sequence
        device: Device to use
    
    Returns:
        best_tokens: Best sequences found (batch_size x seq_len) tensor
    """
    batch_size, seq_len = best_tokens.shape
    
    # Create prefix mask: True for positions that should NOT be modified
    if prefix_lengths is not None:
        # Create a mask where prefix positions are True (protected)
        prefix_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for batch_idx, prefix_len in enumerate(prefix_lengths):
            if prefix_len > 0:
                prefix_mask[batch_idx, :prefix_len] = True
        print(f"  Local search: Protecting prefix tokens (lengths: {prefix_lengths.tolist()})")
    else:
        prefix_mask = None
    
    # NOTE: We calculate best_distances here to initialize batch_top_sequences later
    # This is needed for hierarchical comparison with neighbors
    
    # DEBUG: Log initial best sequence state BEFORE local search
    print(f"  Local search: Initial best sequences (before optimization):")
    
    # Step 1: Decode best_tokens to text (full sequence with prefix)
    # CRITICAL: skip_special_tokens=True to remove <|endoftext|> padding
    best_texts = tokenizer.batch_decode(best_tokens.cpu().numpy(), skip_special_tokens=True)
    
    # Step 2: Clean text by removing special tokens
    best_texts_cleaned = clean_text_samples(best_texts)
    
    # Step 3: For non-perplexity properties, create post-prefix version
    # CRITICAL: NO normalization! File saving uses decode(tokens) directly, NOT encode→decode.
    # Property evaluation must match: decode→clean only, no re-encoding.
    # CRITICAL: skip_special_tokens=True to remove <|endoftext|> padding, matching file saving
    post_prefix_token_ids = remove_prefix_from_token_ids(best_tokens, prefix_lengths)
    post_prefix_texts_raw = tokenizer.batch_decode(post_prefix_token_ids.cpu().numpy(), skip_special_tokens=True)
    post_prefix_texts_cleaned = clean_text_samples(post_prefix_texts_raw)
    
    # Use cleaned text directly without normalization
    best_texts_for_eval = post_prefix_texts_cleaned
    
    # Step 4: Calculate properties
    # For each property, use appropriate text (full vs post-prefix)
    # DEBUG: Print Batch 1 text for investigation (COMMENTED OUT - too verbose)
    # if batch_size > 1:
    #     print(f"    DEBUG Initial eval Batch 1 text (first 200 chars): {repr(best_texts_for_eval[1][:200])}")
    
    best_prop_values = []
    for prop_idx, calc in enumerate(property_calcs_parallel):
        # All properties now use post-prefix text (no prefix)
        prop_value = calc(best_texts_for_eval, batch_size, device)
        best_prop_values.append(prop_value)
    
    # Apply distance functions to get distances
    best_distances = [
        dist_func(prop_value)
        for dist_func, prop_value in zip(distance_to_bounds_parallel, best_prop_values)
    ]
    # Stack into [batch_size, num_properties]
    best_distances_tensor = torch.stack(best_distances, dim=-1)
    
    # DEBUG: Print initial distances AND raw toxicity scores for each batch
    for batch_idx in range(batch_size):
        dist_str = ", ".join([f"{d:.4f}" for d in best_distances_tensor[batch_idx].cpu().tolist()])
        print(f"    Batch {batch_idx} initial distances: [{dist_str}]")
        # DEBUG: Print Batch 2 and 3 with RAW toxicity scores AND FULL TEXT LENGTH (property 0)
        # if batch_idx == 2:
        #     raw_toxicity = best_prop_values[0][batch_idx].item() if len(best_prop_values) > 0 else 0.0
        #     full_text = best_texts_for_eval[batch_idx]
        #     print(f"    DEBUG Batch 2 RAW toxicity score: {raw_toxicity:.6f}")
        #     print(f"    DEBUG Batch 2 text length: {len(full_text)} chars")
        #     print(f"    DEBUG Batch 2 FULL text: {repr(full_text)}")
        # if batch_idx == 3:
        #     raw_toxicity = best_prop_values[0][batch_idx].item() if len(best_prop_values) > 0 else 0.0
        #     full_text = best_texts_for_eval[batch_idx]
        #     print(f"    DEBUG Batch 3 RAW toxicity score: {raw_toxicity:.6f}")
        #     print(f"    DEBUG Batch 3 text length: {len(full_text)} chars")
        #     print(f"    DEBUG Batch 3 FULL text: {repr(full_text)}")
    
    # Get top-k token indices for each position from x_theta using specified sampling method
    # x_theta_probs shape: (batch_size, seq_len, vocab_size)
    topk_indices = select_top_k_tokens(
        x_theta_probs, 
        top_k_values_for_local_search, 
        method=sampling_method, 
        alpha=locally_typical_alpha
    )
    # topk_indices shape: (batch_size, seq_len, top_k_values_for_local_search)
    
    # Print appropriate message based on sampling method
    if sampling_method == 'locally_typical':
        print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens using '{sampling_method}' (additive alpha={locally_typical_alpha}) in parallel...")
    elif sampling_method == 'locally_typical_distance':
        print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens using '{sampling_method}' (scaling alpha={locally_typical_alpha}) in parallel...")
    else:
        print(f"  Local search: generating ALL neighbors from top-{top_k_values_for_local_search} tokens using '{sampling_method}' in parallel...")
    
    # FULLY VECTORIZED neighbor generation across ALL ranks at once!
    # Generate neighbors for ALL ranks simultaneously
    all_neighbor_tokens = []
    all_neighbor_metadata = []
    
    for k_rank in range(0, top_k_values_for_local_search):
        # Extract candidate tokens for this rank: [batch_size, seq_len]
        candidate_tokens = topk_indices[:, :, k_rank]  # Shape: [batch_size, seq_len]
        
        # Create a mask for positions where candidate != current token
        is_different = candidate_tokens != best_tokens  # Shape: [batch_size, seq_len]
        
        # Apply prefix mask: exclude prefix positions from being modified
        if prefix_mask is not None:
            is_different = is_different & (~prefix_mask)
        
        # Get indices of all positions that differ (and are not in prefix)
        batch_indices, pos_indices = torch.where(is_different)
        
        if len(batch_indices) == 0:
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
    
    if len(all_neighbor_tokens) == 0:
        print(f"  Local search: No valid neighbors found, returning original tokens")
        return best_tokens
    
    # Concatenate all neighbors into one big batch
    neighbor_tokens_batch = torch.cat(all_neighbor_tokens, dim=0)
    
    total_neighbors = neighbor_tokens_batch.shape[0]
    print(f"  Local search: Generated {total_neighbors} unique neighbors across all ranks")
    
    print(f"  Local search: Evaluating neighbors with hierarchical property filtering...")
    
    # HIERARCHICAL PROPERTY EVALUATION
    # Instead of computing all properties at once, compute them one by one in priority order
    # After each property, keep only neighbors with minimum distance for that property (per batch)
    # This significantly reduces computation for lower-priority properties
    
    num_properties = len(property_calcs_parallel)
    
    # Initialize distance tensor: will be filled incrementally
    # Shape: [total_neighbors, num_properties]
    neighbor_distances_tensor = torch.zeros((total_neighbors, num_properties), device=device)
    
    # Keep track of active neighbors (indices into neighbor_tokens_batch)
    active_neighbor_indices = list(range(total_neighbors))
    active_neighbor_metadata = list(all_neighbor_metadata)
    
    # Decode texts once (will be reused)
    # CRITICAL: skip_special_tokens=True to remove <|endoftext|> padding
    neighbor_texts = tokenizer.batch_decode(neighbor_tokens_batch.cpu().numpy(), skip_special_tokens=True)
    neighbor_texts_cleaned = clean_text_samples(neighbor_texts)
    
    # Create post-prefix version by slicing token IDs before decoding
    # CRITICAL: Slice token IDs BEFORE decoding to avoid re-tokenization issues
    if prefix_lengths is not None:
        # Create expanded prefix_lengths for all neighbors based on their batch_idx
        expanded_prefix_lengths = torch.tensor([prefix_lengths[metadata[0]].item() 
                                                for metadata in all_neighbor_metadata], 
                                               dtype=torch.long, device=device)
        # Slice token IDs before decoding
        post_prefix_neighbor_tokens = remove_prefix_from_token_ids(neighbor_tokens_batch, expanded_prefix_lengths)
        # CRITICAL: skip_special_tokens=True to remove <|endoftext|> padding
        post_prefix_neighbor_texts_raw = tokenizer.batch_decode(post_prefix_neighbor_tokens.cpu().numpy(), skip_special_tokens=True)
        post_prefix_neighbor_texts_cleaned = clean_text_samples(post_prefix_neighbor_texts_raw)
        
        # CRITICAL: NO normalization! Text evaluation must match text saved to file.
        # File saving uses direct decode(tokens), NOT encode→decode→clean.
        # So property evaluation must use the same: decode→clean, no re-encoding.
        neighbor_texts_for_eval = post_prefix_neighbor_texts_cleaned
    else:
        neighbor_texts_for_eval = neighbor_texts_cleaned
    
    # Process each property in order (priority: first property is most important)
    for prop_idx in range(num_properties):
        if len(active_neighbor_indices) == 0:
            print(f"  Local search: No active neighbors remaining after property {prop_idx-1}")
            break
        
        print(f"  Local search: Property {prop_idx}/{num_properties} - Evaluating {len(active_neighbor_indices)} active neighbors...")
        
        # Get texts for active neighbors only (post-prefix portion)
        active_texts = [neighbor_texts_for_eval[i] for i in active_neighbor_indices]
        
        # Calculate this property for active neighbors
        # All properties now use post-prefix text (no prefix)
        prop_values = property_calcs_parallel[prop_idx](active_texts, len(active_texts), device)
        
        # Apply distance function
        distances = distance_to_bounds_parallel[prop_idx](prop_values)
        
        # Store distances in the full tensor
        for local_idx, global_idx in enumerate(active_neighbor_indices):
            neighbor_distances_tensor[global_idx, prop_idx] = distances[local_idx]
        
        # Debug: print distance distribution for all properties (limit to first 5 to avoid spam)
        if prop_idx < min(5, num_properties):
            # Group by batch_idx for debugging
            batch_neighbor_distances_debug = {b_idx: [] for b_idx in range(batch_size)}
            for local_idx, global_idx in enumerate(active_neighbor_indices):
                b_idx = active_neighbor_metadata[local_idx][0]
                batch_neighbor_distances_debug[b_idx].append(distances[local_idx].item())
            
            for b_idx in range(batch_size):
                if batch_neighbor_distances_debug[b_idx]:
                    distances_this_batch = batch_neighbor_distances_debug[b_idx]
                    min_dist = min(distances_this_batch)
                    count_min = sum(1 for d in distances_this_batch if d == min_dist)
                    unique_dists = len(set(distances_this_batch))
                    print(f"    DEBUG Batch {b_idx} Property {prop_idx}: {len(distances_this_batch)} neighbors, {unique_dists} unique distances, {count_min} at min_dist={min_dist:.6f}")
        
        # If this is NOT the last property, filter neighbors by minimum distance (per batch)
        if prop_idx < num_properties - 1:
            # Group by batch_idx
            batch_neighbor_distances = {b_idx: [] for b_idx in range(batch_size)}
            for local_idx, global_idx in enumerate(active_neighbor_indices):
                b_idx = active_neighbor_metadata[local_idx][0]
                batch_neighbor_distances[b_idx].append((global_idx, distances[local_idx].item()))
            
            # Find minimum distance per batch and keep only those neighbors
            new_active_indices = []
            new_active_metadata = []
            
            for b_idx in range(batch_size):
                if not batch_neighbor_distances[b_idx]:
                    continue
                
                # Find minimum distance for this batch
                min_dist = min(dist for _, dist in batch_neighbor_distances[b_idx])
                
                # Keep neighbors with minimum distance
                for global_idx, dist in batch_neighbor_distances[b_idx]:
                    if dist == min_dist:
                        new_active_indices.append(global_idx)
                        # Find corresponding metadata
                        for i, idx in enumerate(active_neighbor_indices):
                            if idx == global_idx:
                                new_active_metadata.append(active_neighbor_metadata[i])
                                break
            
            filtered_count = len(active_neighbor_indices) - len(new_active_indices)
            active_neighbor_indices = new_active_indices
            active_neighbor_metadata = new_active_metadata
            
            print(f"  Local search: After property {prop_idx}, kept {len(active_neighbor_indices)} neighbors (filtered {filtered_count})")
    
    print(f"  Local search: Final {len(active_neighbor_indices)} neighbors after all property filtering")
    
    print(f"  Local search: Updating best sequences...")
    print(f"  Local search: Comparing {len(active_neighbor_indices)} final neighbors against initial sequences...")
    
    # Track top-k best sequences for each batch element
    # We'll store tuples of (distance_tensor, tokens_tensor) for each sequence
    # Initialize with the original best sequence
    batch_top_sequences = []
    for batch_idx in range(batch_size):
        # Each batch element starts with its current best sequence
        initial_entry = (best_distances_tensor[batch_idx].clone(), best_tokens[batch_idx].clone())
        batch_top_sequences.append([initial_entry])
    
    max_keep = max(best_sequence_rank, 10)
    # Update best sequences if neighbors are better
    # NOTE: Only iterate over ACTIVE neighbors (those that survived hierarchical filtering)
    for local_idx, global_idx in enumerate(active_neighbor_indices):
        batch_idx, pos, k_rank = active_neighbor_metadata[local_idx]
        neighbor_dist = neighbor_distances_tensor[global_idx]  # [num_properties]
        neighbor_tok = neighbor_tokens_batch[global_idx]  # [seq_len]
        
        # Check if this neighbor should be added to top sequences
        top_list = batch_top_sequences[batch_idx]
        
        # Find insertion position using hierarchical comparison
        insert_idx = len(top_list)  # Default: append at end
        neighbor_dist_list = neighbor_dist.cpu().tolist()
        
        for idx, (existing_dist, _) in enumerate(top_list):
            existing_dist_list = existing_dist.cpu().tolist()
            if compare_hierarchical(neighbor_dist_list, existing_dist_list) < 0:
                insert_idx = idx
                break
        
        # Insert neighbor at the appropriate position
        # Keep only top max(best_sequence_rank, 10) to avoid memory issues
        if insert_idx < max_keep:
            top_list.insert(insert_idx, (neighbor_dist.clone(), neighbor_tok.clone()))
            # Trim list to max_keep
            if len(top_list) > max_keep:
                top_list.pop()
    
    # Select the sequence at the specified rank for each batch element
    for batch_idx in range(batch_size):
        top_list = batch_top_sequences[batch_idx]
        # Adjust for 0-indexing (rank 1 = index 0)
        rank_idx = best_sequence_rank - 1
        
        if rank_idx < len(top_list):
            # Use the sequence at the specified rank
            best_distances_tensor[batch_idx] = top_list[rank_idx][0]
            best_tokens[batch_idx] = top_list[rank_idx][1]
        else:
            # If we don't have enough sequences, use the last one available
            best_distances_tensor[batch_idx] = top_list[-1][0]
            best_tokens[batch_idx] = top_list[-1][1]
    
    # DEBUG: Show which sequence was selected (initial vs neighbor)
    print(f"  Local search: Sequence selection summary:")
    for batch_idx in range(batch_size):
        top_list = batch_top_sequences[batch_idx]
        initial_dist = top_list[0][0].cpu().tolist()  # Initial is always at index 0
        selected_dist = best_distances_tensor[batch_idx].cpu().tolist()
        # Format distances dynamically based on number of properties
        initial_str = "[" + ", ".join([f"{d:.2f}" for d in initial_dist]) + "]"
        selected_str = "[" + ", ".join([f"{d:.2f}" for d in selected_dist]) + "]"
        
        # Check if selected is same as initial
        is_same = torch.allclose(top_list[0][0], best_distances_tensor[batch_idx], atol=1e-6)
        source = "KEPT initial" if is_same else "CHOSE neighbor"
        print(f"    Batch {batch_idx}: Initial={initial_str} → Selected={selected_str} ({source})")
    
    if best_sequence_rank > 1:
        print(f"  Local search: Selected sequences with rank {best_sequence_rank} (distance-wise)")
    
    # NO NORMALIZATION! Candidate tokens must stay unchanged!
    # Properties were already calculated consistently during neighbor evaluation
    # Normalization would change the tokens and make properties inconsistent
    print(f"  Local search: Keeping selected tokens unchanged (no normalization)")
    
    # DEBUG: Print final distances (these are from selection, already accurate)
    print(f"  Local search: Final best sequences (after selection):")
    for batch_idx in range(batch_size):
        dist_str = ", ".join([f"{d:.4f}" for d in best_distances_tensor[batch_idx].cpu().tolist()])
        print(f"    Batch {batch_idx} final distances: [{dist_str}]")
    
    # # DEBUG: Print token IDs for batches 2 and 3 to verify what local search is returning
    # print(f"  DEBUG: Local search returning token IDs:")
    # for batch_idx in [2, 3]:
    #     if batch_idx < batch_size:
    #         # Print first 30 token IDs (including prefix)
    #         token_ids_full = best_tokens[batch_idx].cpu().tolist()[:30]
    #         print(f"    Batch {batch_idx} first 30 token IDs: {token_ids_full}")
            
    #         # Decode post-prefix text
    #         if prefix_lengths is not None and batch_idx < len(prefix_lengths):
    #             prefix_len = prefix_lengths[batch_idx].item() if hasattr(prefix_lengths[batch_idx], 'item') else prefix_lengths[batch_idx]
    #             post_prefix_tokens = best_tokens[batch_idx][prefix_len:].cpu().tolist()
    #             # CRITICAL: skip_special_tokens=True to remove <|endoftext|> padding
    #             post_prefix_text = tokenizer.decode(post_prefix_tokens, skip_special_tokens=True).strip()
    #             print(f"    Batch {batch_idx} post-prefix text (from local search): {repr(post_prefix_text[:150])}")
    
    return best_tokens


if __name__ == "__main__":
    print("Language local search utilities for batch processing")
