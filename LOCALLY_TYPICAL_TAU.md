# Locally Typical Sampling with Entropy Scaling (Tau Method)

## Overview

This is an alternative formulation of **Locally Typical Sampling** using **multiplicative entropy scaling** instead of additive probability bias. Both methods use the same parameter `--locally_typical_alpha`, but the interpretation differs based on which method you choose.

## Mathematical Formulas

### Method 1: `locally_typical` (Additive Bias)
```
distance = |-log(p) - H| - alpha * log(p)
```
- **Default alpha**: 0.0 (pure locally typical)
- **Bias direction**: Increase alpha to favor high probability
- **Top-p limit**: alpha → ∞

### Method 2: `locally_typical_distance` (Entropy Scaling)
```
distance = |-log(p) - alpha * H|
```
- **Default alpha**: 1.0 (pure locally typical)  
- **Bias direction**: Decrease alpha to favor high probability
- **Top-p limit**: alpha → 0.0

**Note**: Despite using the same parameter name (`alpha`), the two methods interpret it differently. We recommend setting appropriate defaults based on your choice of method.

## How Alpha Works in Each Method

### For `locally_typical` (Additive Bias):
- **alpha = 0.0**: Pure locally typical
- **alpha > 0.0**: Bias toward high probability (subtract penalty)
- **alpha → ∞**: Approaches top-p

### For `locally_typical_distance` (Entropy Scaling):
- **alpha = 1.0**: Pure locally typical (full entropy target)
- **alpha < 1.0**: Bias toward high probability (scale entropy down)
- **alpha → 0.0**: Approaches top-p (target becomes 0)

**Key Difference**: 
- **Additive method**: Higher alpha = more bias
- **Scaling method**: Lower alpha = more bias

## Comparison: Additive vs Scaling

| Aspect | `locally_typical` | `locally_typical_distance` |
|--------|-------------------|----------------------|
| **Formula** | `\|-log(p) - H\| - α·log(p)` | `\|-log(p) - α·H\|` |
| **Bias Type** | Additive (subtract term) | Multiplicative (scale target) |
| **Default Alpha** | 0.0 (pure locally typical) | 1.0 (pure locally typical) |
| **Increase Bias** | Increase alpha (0→∞) | Decrease alpha (1→0) |
| **Top-p Limit** | α → ∞ | α → 0 |
| **Interpretation** | "Penalty reduction for high prob" | "Scale the entropy target" |

## Intuition

Think of `tau` as an "information target dial":
- **High tau (0.8-1.0)**: Target near full entropy → diverse, exploratory
- **Medium tau (0.4-0.7)**: Balanced → coherent with some diversity
- **Low tau (0.1-0.3)**: Target near zero → conservative, high-prob tokens

The closer tau is to 0, the more the algorithm says:
> "I want tokens with information content close to 0 (i.e., very predictable/high probability tokens)"

## Usage Examples

### Example 1: Additive Method - Pure Locally Typical
```bash
python inference_search.py \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 0.0
```

### Example 2: Additive Method - Bias toward High Prob
```bash
python inference_search.py \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 0.5
```

### Example 3: Scaling Method - Pure Locally Typical
```bash
python inference_search.py \
  --local_search_sampling_method locally_typical_distance \
  --locally_typical_alpha 1.0
```

### Example 4: Scaling Method - Bias toward High Prob
```bash
python inference_search.py \
  --local_search_sampling_method locally_typical_distance \
  --locally_typical_alpha 0.5
```

## Implementation Details

### File: `x_theta_modifier/local_search_language_utils.py`

```python
elif method == 'locally_typical_distance':
    # Calculate entropy
    log_probs = torch.log(x_theta_probs + epsilon)
    entropy = -torch.sum(x_theta_probs * log_probs, dim=-1, keepdim=True)
    
    # Calculate -log(p)
    neg_log_probs = -log_probs
    
    # Scale entropy by tau and compute distance
    scaled_entropy = tau * entropy
    distance = torch.abs(neg_log_probs - scaled_entropy)
    
    # Select tokens with smallest distance
    _, topk_indices = torch.topk(distance, k=top_k, largest=False)
```

## Recommended Tau Values

| Tau | Use Case | Expected Behavior |
|-----|----------|-------------------|
| 1.0 | Pure locally typical | Maximum diversity, may be less coherent |
| 0.7-0.9 | Balanced generation | Good diversity with reasonable coherence |
| 0.5-0.7 | Safe exploration | More coherent, still some diversity |
| 0.3-0.5 | Conservative | Very coherent, limited diversity |
| 0.1-0.3 | Near top-p | Almost identical to highest prob tokens |
| ~0.0 | Top-p equivalent | Pure greedy/top-p behavior |

## Relationship to Information Theory

The formula `distance = |-log(p) - τ·H|` has a clear information-theoretic interpretation:

1. **-log(p)**: Self-information (surprise) of token
2. **H**: Expected self-information (entropy)
3. **τ·H**: Scaled target information level

When τ = 1.0, we select tokens whose surprise matches the expected surprise (typical tokens).
When τ < 1.0, we lower the target, preferring less surprising (more predictable) tokens.

## Testing

Run the test script to compare different tau values:

```bash
bash test_locally_typical_distance.sh
```

This will generate samples with:
- tau=1.0 (pure locally typical)
- tau=0.7 (slight bias)
- tau=0.5 (moderate bias)
- tau=0.2 (strong bias)
- alpha=0.5 (for comparison with additive method)
- top_p (for comparison)

## When to Use Which Method?

### Use `locally_typical_distance` when:
- You want intuitive entropy scaling (what % of full entropy?)
- You prefer multiplicative rather than additive bias
- Your use case has a natural interpretation of "information target"

### Use `locally_typical` (alpha) when:
- You want to directly penalize high-prob tokens additively
- You want unbounded bias control (alpha can be > 1)
- You prefer thinking in terms of "how much extra weight for probability"

Both methods achieve similar goals but with different parameterizations. Experiment with both to see which works better for your use case!

## Related Parameters

- `--local_search_sampling_method`: Choose 'top_p', 'locally_typical', or 'locally_typical_distance'
- `--locally_typical_distance`: Set tau value (default: 1.0)
- `--locally_typical_alpha`: Set alpha value for 'locally_typical' method (default: 0.0)
- `--top_k_values_for_local_search`: Number of tokens to select (default: 10)

## References

- Meister, C., et al. (2022). "Locally Typical Sampling". arXiv:2202.00666
- This `locally_typical_distance` variant provides an alternative parameterization with entropy scaling
