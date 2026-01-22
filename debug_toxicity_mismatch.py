#!/usr/bin/env python3
"""
Debug script to test toxicity calculation mismatch.
Compares property calculation vs evaluation script results.
"""

import torch
import sys
sys.path.insert(0, '/bigtemp/nzj6jt/workspace/project/mdlm_llm_search')

from properties.toxicity_property import calc_toxicity_parallel
from evaluation.toxicity import load_model, tokenizer

# Read the generated text from file 487
file_path = "sample_results/openwebtext-split/topk_all_nucleus_all/standard/local_search_language/['measure_combined_heuristic', 'toxicity', 'perplexity']_lb[-100.0, -100.0, -100.0]_ub[0.0, 0.25, 0.0]/rank_1/top_p/0.0/local_8_candidates_numtheta_8_length250/seed_0/molecules/487.txt"

with open(file_path, 'r') as f:
    generated_text = f.read().strip()

print("=" * 80)
print("GENERATED TEXT (from file):")
print("=" * 80)
print(generated_text)
print(f"\nLength: {len(generated_text)} chars, {len(generated_text.split())} words")
print()

# Test 1: Using property calculation (calc_toxicity_parallel)
print("=" * 80)
print("TEST 1: Property Calculation (calc_toxicity_parallel)")
print("=" * 80)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
toxicity_scores = calc_toxicity_parallel([generated_text], 1, device, max_length=100)
print(f"Toxicity score: {toxicity_scores[0].item():.6f}")
print(f"Distance to 0.25: {max(0, toxicity_scores[0].item() - 0.25):.6f}")
print()

# Test 2: Using evaluation script (same as toxicity.py)
print("=" * 80)
print("TEST 2: Evaluation Script (toxicity.py method)")
print("=" * 80)
model = load_model('outputs/toxicity/classifier/toxicity_gpt2_neo.pt')
inputs = tokenizer([generated_text], return_tensors="pt", padding=True, truncation=True, max_length=100)
input_ids = inputs["input_ids"].to(model.gpt_neo.device)
attention_mask = inputs["attention_mask"].to(model.gpt_neo.device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

eval_score = outputs['logits'].cpu().item()
print(f"Toxicity score: {eval_score:.6f}")
print(f"Distance to 0.25: {max(0, eval_score - 0.25):.6f}")
print()

# Test 3: Check tokenization differences
print("=" * 80)
print("TEST 3: Tokenization Analysis")
print("=" * 80)
from transformers import GPT2Tokenizer

# Get tokenizer used in properties
prop_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
prop_tokenizer.pad_token = prop_tokenizer.eos_token

# Tokenize
prop_tokens = prop_tokenizer(generated_text, return_tensors="pt", padding=True, truncation=True, max_length=100)
eval_tokens = tokenizer(generated_text, return_tensors="pt", padding=True, truncation=True, max_length=100)

print(f"Property tokenizer num tokens: {prop_tokens['input_ids'].shape[1]}")
print(f"Eval tokenizer num tokens: {eval_tokens['input_ids'].shape[1]}")
print(f"Tokens match: {torch.equal(prop_tokens['input_ids'], eval_tokens['input_ids'])}")

# Decode back
prop_decoded = prop_tokenizer.decode(prop_tokens['input_ids'][0], skip_special_tokens=True)
eval_decoded = tokenizer.decode(eval_tokens['input_ids'][0], skip_special_tokens=True)

print(f"\nProperty tokenizer decoded: {prop_decoded[:100]}...")
print(f"Eval tokenizer decoded: {eval_decoded[:100]}...")
print(f"Decoded text matches: {prop_decoded == eval_decoded}")

# Check if truncation happened
print(f"\nOriginal text length: {len(generated_text)}")
print(f"Property decoded length: {len(prop_decoded)}")
print(f"Eval decoded length: {len(eval_decoded)}")
print(f"Text was truncated: {len(prop_decoded) < len(generated_text)}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Property calculation: {toxicity_scores[0].item():.6f}")
print(f"Evaluation script:    {eval_score:.6f}")
print(f"Difference:           {abs(eval_score - toxicity_scores[0].item()):.6f}")
print(f"Threshold:            0.25")
print(f"Property passes:      {toxicity_scores[0].item() <= 0.25}")
print(f"Evaluation passes:    {eval_score <= 0.25}")
print()

if abs(eval_score - toxicity_scores[0].item()) > 0.001:
    print("⚠️  MISMATCH DETECTED!")
    print("The two methods produce different toxicity scores.")
else:
    print("✓ Results match!")
