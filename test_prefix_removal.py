#!/usr/bin/env python3
"""Test the new prefix removal logic to ensure it works correctly."""

import torch
from transformers import GPT2Tokenizer

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Test case: Create a sequence with prefix
prefix_text = "Adolescents who are exposed to more sexual content in movies start h"
generated_text = "aving longer term sexual partners"
full_text = prefix_text + generated_text

print("=" * 80)
print("TEST: Prefix Removal Logic")
print("=" * 80)
print(f"Prefix: {prefix_text}")
print(f"Generated: {generated_text}")
print(f"Full: {full_text}")
print()

# Tokenize full sequence
full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
print(f"Full tokens: {len(full_tokens)} tokens")
print(f"Token IDs: {full_tokens[:10]}... (first 10)")
print()

# Tokenize prefix to get length
prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
prefix_len = len(prefix_tokens)
print(f"Prefix length: {prefix_len} tokens")
print()

# Method 1: OLD METHOD (decode then re-encode - WRONG!)
print("=" * 80)
print("METHOD 1: Decode full → Remove prefix by re-encoding (OLD - BUGGY)")
print("=" * 80)
decoded_full = tokenizer.decode(full_tokens, skip_special_tokens=True)
print(f"Decoded full: {decoded_full}")

# Re-encode to get tokens
reencoded_tokens = tokenizer.encode(decoded_full, add_special_tokens=False)
print(f"Re-encoded: {len(reencoded_tokens)} tokens (was {len(full_tokens)})")

# Try to remove prefix by slicing re-encoded tokens
post_prefix_tokens_wrong = reencoded_tokens[prefix_len:]
post_prefix_text_wrong = tokenizer.decode(post_prefix_tokens_wrong)
print(f"Post-prefix text (WRONG): {post_prefix_text_wrong}")
print()

# Method 2: NEW METHOD (slice token IDs then decode - CORRECT!)
print("=" * 80)
print("METHOD 2: Slice token IDs → Decode (NEW - CORRECT)")
print("=" * 80)
# Slice token IDs directly
post_prefix_tokens_correct = full_tokens[prefix_len:]
print(f"Post-prefix tokens: {len(post_prefix_tokens_correct)} tokens")
print(f"Token IDs: {post_prefix_tokens_correct}")

# Decode
post_prefix_text_correct = tokenizer.decode(post_prefix_tokens_correct)
print(f"Post-prefix text (CORRECT): {post_prefix_text_correct}")
print()

# Verify
print("=" * 80)
print("VERIFICATION")
print("=" * 80)
print(f"Expected generated text: {generated_text}")
print(f"Method 1 result: {post_prefix_text_wrong}")
print(f"Method 2 result: {post_prefix_text_correct}")
print(f"Method 1 matches: {post_prefix_text_wrong.strip() == generated_text.strip()}")
print(f"Method 2 matches: {post_prefix_text_correct.strip() == generated_text.strip()}")
