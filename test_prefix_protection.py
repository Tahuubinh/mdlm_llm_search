"""
Test script to verify prefix protection in local search.
"""

import torch
from x_theta_modifier.local_search_language_utils import local_search_language_batch


class DummyTokenizer:
    """Dummy tokenizer for testing"""
    def batch_decode(self, token_ids):
        # Simple decode: convert token IDs to strings
        # Handle both tensor and numpy array
        if hasattr(token_ids, 'tolist'):
            # numpy array
            return [" ".join(map(str, ids)) for ids in token_ids.tolist()]
        else:
            # tensor
            return [" ".join(map(str, ids.cpu().tolist())) for ids in token_ids]
    
    def encode(self, text, add_special_tokens=False):
        # Simple encode: convert string back to token IDs
        return [int(x) for x in text.split()]
    
    def decode(self, token_ids):
        # Simple decode: convert token IDs to string
        return " ".join(map(str, token_ids))


def dummy_property_calc(texts, batch_size, device):
    """Dummy property calculator"""
    # Return random scores
    return torch.rand(batch_size, device=device)


def dummy_distance_func(prop_values):
    """Dummy distance function"""
    return prop_values  # Just pass through


def test_prefix_protection():
    """Test that prefix tokens are not modified during local search"""
    print("\n=== Testing Prefix Protection ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    top_k = 5
    
    # Create dummy data
    best_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    x_theta_probs = torch.rand(batch_size, seq_len, vocab_size, device=device)
    x_theta_probs = x_theta_probs / x_theta_probs.sum(dim=-1, keepdim=True)  # Normalize
    
    # Prefix lengths: first sequence has 3 tokens, second has 5 tokens
    prefix_lengths = torch.tensor([3, 5], dtype=torch.long, device=device)
    
    # Save original prefix tokens
    original_prefixes = best_tokens.clone()
    print(f"Original tokens (batch 0): {best_tokens[0].cpu().tolist()}")
    print(f"Original tokens (batch 1): {best_tokens[1].cpu().tolist()}")
    print(f"Prefix lengths: {prefix_lengths.tolist()}")
    print()
    
    # Run local search
    tokenizer = DummyTokenizer()
    property_calcs_parallel = [dummy_property_calc]
    distance_to_bounds_parallel = [dummy_distance_func]
    
    result_tokens = local_search_language_batch(
        best_tokens=best_tokens,
        x_theta_probs=x_theta_probs,
        distance_to_bounds_parallel=distance_to_bounds_parallel,
        property_calcs_parallel=property_calcs_parallel,
        tokenizer=tokenizer,
        top_k_values_for_local_search=top_k,
        sampling_method='top_p',
        locally_typical_alpha=0.0,
        best_sequence_rank=1,
        prefix_lengths=prefix_lengths,
        device=device
    )
    
    print(f"\nResult tokens (batch 0): {result_tokens[0].cpu().tolist()}")
    print(f"Result tokens (batch 1): {result_tokens[1].cpu().tolist()}")
    print()
    
    # Verify prefix tokens are unchanged
    for batch_idx in range(batch_size):
        prefix_len = prefix_lengths[batch_idx].item()
        original_prefix = original_prefixes[batch_idx, :prefix_len]
        result_prefix = result_tokens[batch_idx, :prefix_len]
        
        if torch.all(original_prefix == result_prefix):
            print(f"✓ Batch {batch_idx}: Prefix (length {prefix_len}) is protected")
        else:
            print(f"✗ Batch {batch_idx}: Prefix (length {prefix_len}) was MODIFIED!")
            print(f"  Original: {original_prefix.cpu().tolist()}")
            print(f"  Result:   {result_prefix.cpu().tolist()}")
    
    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    test_prefix_protection()
