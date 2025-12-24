# perplexity.py
# Calculate perplexity for generated sequences using GPT-2

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import math
import numpy as np


def load_gpt2_model(model_name='gpt2-large'):
    """Load GPT-2 model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def calculate_perplexity(model, tokenizer, text, device):
    """Calculate perplexity for a single text - matching the reference implementation"""
    if not text or len(text.strip()) == 0:
        return float('inf')
    
    # Tokenize the text (no truncation, padding="longest" for single text)
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,  # Don't truncate
        padding="longest",  # Padding style from reference code
    )
    
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    # Calculate perplexity using the same method as reference code
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        nll = outputs.loss.item()  # This is the average cross-entropy over tokens
    
    ppl = math.exp(nll)
    return ppl


def calculate_perplexity_batch(model, tokenizer, texts, device, batch_size=50):
    """Calculate perplexity for a batch of texts"""
    perplexities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the batch
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        # Calculate perplexity for each sequence in the batch
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Get per-sample loss
            # Reshape logits and labels for per-token loss calculation
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            
            # Calculate loss per sample
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss.view(shift_labels.size())

            # Apply attention mask and calculate mean loss per sample
            loss = (loss * shift_attention_mask).sum(dim=1) / shift_attention_mask.sum(dim=1)
            
            # Convert to perplexity
            batch_perplexities = torch.exp(loss).cpu().numpy()
            perplexities.extend(batch_perplexities)
    
    return perplexities


if __name__ == "__main__":
    # Configuration
    output_sequence_dir = "sample_results/openwebtext-split/topk_all_nucleus_all/standard/bon/['toxicity', 'perplexity']_lb[-100.0, -100.0]_ub[0.75, 0.0]/test/seed_0/molecules/"
    num_samples = 8
    batch_size = 8
    
    # Load GPT-2 model (using 'gpt2-large' as per the paper)
    print("Loading GPT-2 model...")
    model, tokenizer, device = load_gpt2_model('gpt2-large')
    print("Model loaded successfully!\n")
    
    # Read all text files
    print(f"Reading {num_samples} files...")
    texts = []
    valid_indices = []
    
    for i in range(num_samples):
        file_path = os.path.join(output_sequence_dir, f"{i}.txt")
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist.")
            texts.append("")
        else:
            with open(file_path, 'r') as f:
                text = f.read().strip()
                texts.append(text)
                if text:  # Only count non-empty texts
                    valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} valid files.\n")
    
    # Calculate perplexity for all samples - USE SIMPLE METHOD (one at a time)
    print("Calculating perplexities...")
    perplexities = []
    for i, text in enumerate(texts):
        if text:
            ppl = calculate_perplexity(model, tokenizer, text, device)
            perplexities.append(ppl)
            if i < 10:  # Print first 10 for debugging
                print(f"Sample {i}: Perplexity = {ppl:.4f}, Text length = {len(text.split())} words")
        else:
            perplexities.append(float('inf'))
    
    # Filter out invalid perplexities (from empty texts)
    valid_perplexities = [perplexities[i] for i in valid_indices]
    
    # Print results for each file
    print("\n" + "="*80)
    print("PERPLEXITY RESULTS")
    print("="*80)
    
    for i in range(num_samples):
        file_path = os.path.join(output_sequence_dir, f"{i}.txt")
        ppl = perplexities[i]
        
        if not texts[i]:
            print(f"File {i}: EMPTY - Perplexity: N/A")
        else:
            print(f"File {i}: Perplexity = {ppl:.4f}")
            print(f"  Text preview: {texts[i][:100]}...")
            print("---")
    
    # Calculate and print statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total files: {num_samples}")
    print(f"Valid files: {len(valid_indices)}")
    print(f"Empty files: {num_samples - len(valid_indices)}")
    
    if valid_perplexities:
        print(f"\nPerplexity Statistics:")
        print(f"  Mean: {np.mean(valid_perplexities):.4f}")
        print(f"  Median: {np.median(valid_perplexities):.4f}")
        print(f"  Std Dev: {np.std(valid_perplexities):.4f}")
        print(f"  Min: {np.min(valid_perplexities):.4f}")
        print(f"  Max: {np.max(valid_perplexities):.4f}")
        
        # Calculate percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = np.percentile(valid_perplexities, p)
            print(f"  {p}th percentile: {value:.4f}")
        
        # Count samples with low perplexity (good quality)
        thresholds = [10, 20, 50, 100]
        print(f"\nPerplexity Distribution:")
        for threshold in thresholds:
            count = sum(1 for ppl in valid_perplexities if ppl < threshold)
            percentage = (count / len(valid_perplexities)) * 100
            print(f"  Perplexity < {threshold}: {count} samples ({percentage:.2f}%)")
