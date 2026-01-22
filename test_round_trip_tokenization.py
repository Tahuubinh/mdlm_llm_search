#!/usr/bin/env python3
"""Test if decode→encode round trip preserves token IDs."""

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Test with sample text
original_tokens = [2782, 2149, 3381, 3206, 4887, 287, 257, 1864, 284, 511, 1714, 13]
print("Original token IDs:", original_tokens)

# Decode
decoded_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
print(f"Decoded text: '{decoded_text}'")

# Re-encode
reencoded_tokens = tokenizer.encode(decoded_text, add_special_tokens=False)
print("Re-encoded token IDs:", reencoded_tokens)

print(f"\nTokens match: {original_tokens == reencoded_tokens}")
if original_tokens != reencoded_tokens:
    print("⚠️  MISMATCH! Decode→encode changes tokens!")
    print(f"Original length: {len(original_tokens)}")
    print(f"Re-encoded length: {len(reencoded_tokens)}")
else:
    print("✓ Round trip preserves tokens")
