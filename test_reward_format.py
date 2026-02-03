"""
Test different prompt formats for OpenAssistant reward model
to find the correct one that gives scores in -10 to +10 range
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Disable TorchScript
torch.jit._state.disable()

model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
model.to('cuda')

# Test texts
good_text = "He has seen this and realized the importance of racial equality."
bad_text = "he has seen this and have the and the of and and of..."

print("="*80)
print("Testing different prompt formats for OpenAssistant Reward Model")
print("="*80)

# Format 1: Raw text (no template)
print("\n1. RAW TEXT (no template):")
for label, text in [("Good", good_text), ("Bad", bad_text)]:
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    with torch.no_grad():
        score = model(**inputs).logits[0][0].item()
    print(f"   {label}: {score:.4f}")

# Format 2: With <|prompter|> and <|assistant|> tags
print("\n2. WITH <|prompter|> and <|assistant|> tags:")
for label, text in [("Good", good_text), ("Bad", bad_text)]:
    formatted = f"<|prompter|>Evaluate the quality of this text.<|endoftext|><|assistant|>{text}<|endoftext|>"
    inputs = tokenizer(formatted, return_tensors="pt").to('cuda')
    with torch.no_grad():
        score = model(**inputs).logits[0][0].item()
    print(f"   {label}: {score:.4f}")

# Format 3: Question-Answer format
print("\n3. QUESTION-ANSWER format:")
for label, text in [("Good", good_text), ("Bad", bad_text)]:
    formatted = f"Question: What is the quality of this text?\n\nAnswer: {text}"
    inputs = tokenizer(formatted, return_tensors="pt").to('cuda')
    with torch.no_grad():
        score = model(**inputs).logits[0][0].item()
    print(f"   {label}: {score:.4f}")

# Format 4: Check if tokenizer has special tokens
print("\n4. TOKENIZER SPECIAL TOKENS:")
print(f"   Special tokens: {tokenizer.special_tokens_map}")
print(f"   All special tokens: {tokenizer.all_special_tokens}")
print(f"   Vocab size: {len(tokenizer)}")

# Format 5: Try simple conversation format
print("\n5. SIMPLE CONVERSATION format:")
for label, text in [("Good", good_text), ("Bad", bad_text)]:
    formatted = f"User: Rate this text.\nAssistant: {text}"
    inputs = tokenizer(formatted, return_tensors="pt").to('cuda')
    with torch.no_grad():
        score = model(**inputs).logits[0][0].item()
    print(f"   {label}: {score:.4f}")

print("\n" + "="*80)
print("Expected: Good text should have HIGHER score than Bad text")
print("Expected range: Typically -10 to +10 for reward models")
print("="*80)
