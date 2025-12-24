#!/bin/bash
# Test script for BoN with toxicity property on openwebtext-split

echo "Testing BoN with toxicity property..."

python inference_search.py \
    --data openwebtext-split \
    --num_samples 5 \
    --batch_size 5 \
    --version test_bon_toxicity \
    --seed 42 \
    --posterior_sampling_method standard \
    --num_posterior_samples 32 \
    --x_theta_type bon \
    --num_x_theta_samples 8 \
    --prefix_dir data/toxicity/1000_samples \
    --argmax_mode none \
    --property_type toxicity \
    --lower_bound 0.0 \
    --upper_bound 0.3

echo "Done! Check output in sample_results/openwebtext-split/"
