#!/bin/bash
# Test to verify best sequence is preserved across steps

python inference_search.py \
  --data openwebtext-split \
  --prefix_dir data/toxicity/1000_samples \
  --num_samples 8 \
  --batch_size 8 \
  --version test_best_preserved \
  --x_theta_type local_search_language \
  --lower_bound -100 -100 -100 \
  --upper_bound 0 0.25 0 \
  --property_type measure_combined_heuristic toxicity perplexity \
  --seed 0 \
  --top_k_values_for_local_search 4 \
  --num_x_theta_samples 8 \
  --local_search_sampling_method top_p \
  --best_sequence_rank 1 \
  --diffusion_steps 10 \
  --start_sample_index 200 2>&1 | grep -E "(Step [0-9]+:|DEBUG|Batch 6)"
