#!/bin/bash
# Test locally_typical_distance (entropy scaling) with different alpha values
# Note: For this method, alpha scales the entropy target:
#   alpha = 1.0: pure locally typical (target = H)
#   alpha < 1.0: scale entropy down → bias toward high probability
#   alpha → 0.0: target → 0 → top-p behavior

# Example 1: Pure locally typical (alpha=1.0)
echo "Testing alpha=1.0 (pure locally typical, entropy scaling)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version tau_alpha_1.0 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical_distance \
  --locally_typical_alpha 1.0

# Example 2: Slight bias toward high prob (alpha=0.7)
echo "Testing alpha=0.7 (slight bias, entropy scaling)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version tau_alpha_0.7 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical_distance \
  --locally_typical_alpha 0.7

# Example 3: Moderate bias (alpha=0.5)
echo "Testing alpha=0.5 (moderate bias, entropy scaling)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version tau_alpha_0.5 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical_distance \
  --locally_typical_alpha 0.5

# Example 4: Strong bias (alpha=0.2) - closer to top-p
echo "Testing alpha=0.2 (strong bias, approaching top-p)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version tau_alpha_0.2 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical_distance \
  --locally_typical_alpha 0.2

# Example 5: Compare with locally_typical (alpha=0.5, additive bias)
echo "Testing locally_typical with alpha=0.5 (additive bias, for comparison)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version additive_alpha_0.5 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 0.5
