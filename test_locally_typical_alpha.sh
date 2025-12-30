#!/bin/bash
# Test locally typical sampling with different alpha values
# alpha controls the bias toward high probability tokens:
#   alpha = 0.0: pure locally typical (entropy-based only)
#   alpha > 0.0: bias toward high probability tokens
#   alpha → ∞: approaches top-p behavior

# Example 1: Pure locally typical (alpha=0.0)
echo "Testing alpha=0.0 (pure locally typical)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version alpha_0.0 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 0.0

# Example 2: Small bias toward high prob (alpha=0.5)
echo "Testing alpha=0.5 (slight bias toward high probability)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version alpha_0.5 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 0.5

# Example 3: Moderate bias (alpha=1.0)
echo "Testing alpha=1.0 (moderate bias toward high probability)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version alpha_1.0 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 1.0

# Example 4: Strong bias (alpha=2.0) - closer to top-p
echo "Testing alpha=2.0 (strong bias, approaching top-p)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version alpha_2.0 \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method locally_typical \
  --locally_typical_alpha 2.0

# Example 5: For comparison - pure top_p
echo "Testing top_p (for comparison)..."
python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --start_sample_index 0 \
  --num_samples 2 \
  --batch_size 2 \
  --version top_p \
  --x_theta_type local_search_language \
  --lower_bound -100 15 \
  --upper_bound 0.75 30 \
  --property_type toxicity perplexity \
  --seed 0 \
  --num_x_theta_samples 1 \
  --top_k_values_for_local_search 5 \
  --local_search_sampling_method top_p
