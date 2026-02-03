"""
Evaluate text quality using OpenAssistant reward model.
Reward model: OpenAssistant/reward-model-deberta-v3-large-v2
Returns quality scores (higher = better quality text)
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_reward_model():
    """
    Load OpenAssistant reward model for text quality evaluation.
    Returns scores where higher = better quality.
    """
    print("Loading OpenAssistant reward model...")
    model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Reward model loaded on {device}")
    return model, tokenizer, device


def evaluate_reward_batch(texts, model, tokenizer, device, batch_size=16):
    """
    Evaluate text quality using reward model.
    Returns reward scores (higher = better).
    """
    scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Reward Model"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Get reward scores
        with torch.no_grad():
            outputs = model(**inputs)
            # Reward model outputs logits, we take the score directly
            reward_scores = outputs.logits.squeeze(-1).cpu().numpy()
            
            # Handle single sample case
            if reward_scores.ndim == 0:
                reward_scores = [float(reward_scores)]
            else:
                reward_scores = reward_scores.tolist()
            
            scores.extend(reward_scores)
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate text quality using OpenAssistant reward model')
    parser.add_argument('--base-path', type=str, 
                    default="sample_results/openwebtext-split/topk_all_nucleus_all/standard/local_search_language/['toxicity', 'measure_combined_heuristic', 'perplexity']_lb[-100.0, -100.0, -100.0]_ub[0.5, 0.0, 0.0]/rank_1/top_p/0.0/local_5_candidates_numtheta_8_length100/seed_1/molecules",
                    help='Path to directory containing text files to evaluate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--output-name', type=str, default='reward_scores.json',
                        help='Output JSON filename')
    args = parser.parse_args()

    # Load text files
    base_path = args.base_path
    if not os.path.exists(base_path):
        print(f"ERROR: Path does not exist: {base_path}")
        exit(1)

    all_files = os.listdir(base_path)
    txt_files = sorted([os.path.join(base_path, f) for f in all_files if f.endswith('.txt')])
    print(f"Found {len(txt_files)} text files in {base_path}")

    # Read all sentences
    sentences = []
    file_names = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            sentences.append(content)
            file_names.append(os.path.basename(txt_file))

    print(f"Loaded {len(sentences)} sentences\n")

    # Load reward model
    print("="*80)
    print("Evaluating with OpenAssistant Reward Model")
    print("="*80)
    model, tokenizer, device = load_reward_model()
    
    # Evaluate
    reward_scores = evaluate_reward_batch(
        sentences, 
        model, 
        tokenizer, 
        device,
        batch_size=args.batch_size
    )
    
    # Build results
    results = []
    for file_name, sentence, score in zip(file_names, sentences, reward_scores):
        results.append({
            "file": file_name,
            "sentence": sentence,
            "reward_score": float(score)
        })
    
    # Print statistics
    print(f"\nReward Score Statistics:")
    print(f"  Mean: {np.mean(reward_scores):.4f}")
    print(f"  Std: {np.std(reward_scores):.4f}")
    print(f"  Min: {np.min(reward_scores):.4f}")
    print(f"  Max: {np.max(reward_scores):.4f}")
    
    # Save results
    output_dir = os.path.join(base_path, "..")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
