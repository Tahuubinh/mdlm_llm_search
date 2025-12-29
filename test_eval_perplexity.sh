#!/bin/bash
# Example script to evaluate perplexity of generated samples

# Example 1: Evaluate samples 0-7 (8 samples)
python evaluation/perplexity.py \
  --output_sequence_dir "sample_results/openwebtext-split/topk_all_nucleus_all/standard/local_search_language/['toxicity', 'perplexity']_lb[-100.0, 20.0]_ub[0.75, 30.0]/8_test_top_5_100steps/seed_0/molecules/" \
  --start_sample_index 0 \
  --num_samples 8 \
  --batch_size 8 \
  --model_name gpt2-large

# Example 2: Evaluate samples 10-29 (20 samples starting from index 10)
# python evaluation/perplexity.py \
#   --output_sequence_dir "sample_results/openwebtext-split/.../molecules/" \
#   --start_sample_index 10 \
#   --num_samples 20 \
#   --batch_size 8 \
#   --model_name gpt2-large
