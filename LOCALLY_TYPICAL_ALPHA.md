# Locally Typical Sampling with Probability Bias (Alpha)

## Overview

This implementation extends **Locally Typical Sampling** with a probability bias parameter `alpha` that allows you to smoothly interpolate between pure entropy-based sampling and top-p (highest probability) sampling.

## Mathematical Formula

The token selection distance is computed as:

```
distance = |(-log(p)) - H| - alpha * log(p)
```

Where:
- `p` = token probability
- `H` = entropy of the distribution = -Σ p*log(p)
- `alpha` = probability bias weight (≥ 0)

Tokens with the **smallest distance** are selected.

## How Alpha Works

### Alpha = 0.0 (Pure Locally Typical)
- **Formula**: `distance = |-log(p) - H|`
- **Behavior**: Selects tokens whose information content (`-log(p)`) is closest to the distribution's entropy
- **Effect**: Balances between predictability and diversity
- **Use case**: Avoiding mode collapse while maintaining coherent text

### Alpha > 0.0 (Bias toward High Probability)
- **Formula**: `distance = |-log(p) - H| - alpha * log(p)`
- **Behavior**: The `-alpha * log(p)` term reduces distance for high-probability tokens
- **Effect**: Higher `alpha` → stronger preference for high-probability tokens
- **Use case**: When you want more "safe" choices but still avoid pure top-p

### Alpha → ∞ (Approaches Top-P)
- **Formula**: `-alpha * log(p)` term dominates
- **Behavior**: Essentially becomes `distance ≈ -alpha * log(p)`, selecting highest probability tokens
- **Effect**: Nearly identical to top-p sampling
- **Use case**: Maximum coherence, similar to top-p

## Intuition

Think of `alpha` as a "safety knob":
- **Low alpha (0-0.5)**: Adventurous, explores diverse tokens near entropy
- **Medium alpha (0.5-1.5)**: Balanced, prefers likely tokens while considering entropy
- **High alpha (1.5+)**: Conservative, strongly favors high-probability tokens

## Usage Examples

### Example 1: Pure Locally Typical
```bash
python inference_search.py \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 0.0 \
  --top_k_values_for_local_search 5
```
Output: Most diverse, avoids mode collapse

### Example 2: Slight Bias
```bash
python inference_search.py \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 0.5 \
  --top_k_values_for_local_search 5
```
Output: Diverse but slightly safer choices

### Example 3: Strong Bias
```bash
python inference_search.py \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 2.0 \
  --top_k_values_for_local_search 5
```
Output: Close to top-p, very coherent

### Example 4: Top-P Comparison
```bash
python inference_search.py \
  --local_search_sampling_method top_p \
  --top_k_values_for_local_search 5
```
Output: Pure top-p, highest probability tokens only

## Implementation Details

### File: `x_theta_modifier/local_search_language_utils.py`

```python
def select_top_k_tokens(x_theta_probs, top_k, method='top_p', alpha=0.0):
    if method == 'locally_typical':
        # Calculate entropy
        log_probs = torch.log(x_theta_probs + epsilon)
        entropy = -torch.sum(x_theta_probs * log_probs, dim=-1, keepdim=True)
        
        # Calculate base distance to entropy
        neg_log_probs = -log_probs
        distance_to_entropy = torch.abs(neg_log_probs - entropy)
        
        # Apply probability bias
        if alpha > 0:
            distance_to_entropy = distance_to_entropy - alpha * log_probs
        
        # Select tokens with smallest distance
        _, topk_indices = torch.topk(distance_to_entropy, k=top_k, largest=False)
        return topk_indices
```

## Recommended Alpha Values

Based on experimentation:

| Alpha | Use Case | Expected Behavior |
|-------|----------|-------------------|
| 0.0 | Avoid mode collapse | Most diverse, may be less coherent |
| 0.3-0.5 | Balanced generation | Good diversity with reasonable coherence |
| 0.8-1.2 | Safe exploration | Coherent with some diversity |
| 1.5-2.0 | Near top-p | Very coherent, close to highest prob tokens |
| 3.0+ | Essentially top-p | Identical to top-p sampling |

## Testing

Run the test script to compare different alpha values:

```bash
bash test_locally_typical_alpha.sh
```

This will generate samples with:
- alpha=0.0 (pure locally typical)
- alpha=0.5 (slight bias)
- alpha=1.0 (moderate bias)
- alpha=2.0 (strong bias)
- top_p (for comparison)

Compare the outputs to find the optimal alpha for your use case.

## Related Parameters

- `--local_search_sampling_method`: Choose 'top_p' or 'locally_typical'
- `--locally_typical_alpha`: Set alpha value (default: 0.0)
- `--top_k_values_for_local_search`: Number of tokens to select (default: 10)

## References

- Meister, C., et al. (2022). "Locally Typical Sampling". arXiv:2202.00666
- Original implementation focuses on pure locally typical (alpha=0)
- This extension adds probability bias for practical flexibility
