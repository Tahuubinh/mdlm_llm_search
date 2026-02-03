"""
Evaluate text coherence and grammaticality using specialized models:
1. UniEval (T5-Large) - for coherence evaluation
2. DeBERTa-v3-Large (CoLA) - for grammatical acceptability
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Try to load transformers first
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    print(f"Error: Missing transformers package - {e}")
    print("Please install: pip install transformers torch")
    exit(1)

# Try to import UniEval - handle different possible locations
UNIEVAL_AVAILABLE = False
unieval_evaluator = None

# Try multiple possible UniEval locations
possible_paths = [
    os.path.join(project_root, 'UniEval'),  # In project root
    os.path.join(project_root, '..', 'UniEval'),  # In parent directory
]

for unieval_path in possible_paths:
    if os.path.exists(unieval_path) and unieval_path not in sys.path:
        sys.path.insert(0, unieval_path)
        try:
            from metric.evaluator import get_evaluator
            UNIEVAL_AVAILABLE = True
            print(f"Successfully imported UniEval from: {unieval_path}")
            break
        except ImportError as e:
            # Remove from path if import failed
            if unieval_path in sys.path:
                sys.path.remove(unieval_path)
            continue

if not UNIEVAL_AVAILABLE:
    print("=" * 80)
    print("WARNING: UniEval is not available.")
    print("To enable UniEval, please:")
    print("  1. Clone the repository:")
    print("     git clone https://github.com/maszhongming/UniEval.git")
    print("  2. Make sure it's in your project root directory")
    print("=" * 80)
    print("DeBERTa-CoLA model will still be available.\n")


def load_deberta_cola():
    """
    Load RoBERTa fine-tuned on CoLA for grammatical acceptability.
    Returns scores between 0-1 (higher = more grammatical).
    """
    print("Loading RoBERTa-base (CoLA) model...")
    model_name = "textattack/roberta-base-CoLA"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"CoLA model loaded on {device}")
    return model, tokenizer, device


def evaluate_cola_batch(texts, model, tokenizer, device, batch_size=16):
    """
    Evaluate grammatical acceptability using DeBERTa-CoLA.
    Returns scores between 0-1 (higher = more acceptable).
    """
    scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="DeBERTa-CoLA"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            # Get probability of acceptable class (usually index 1)
            acceptable_probs = probs[:, 1].cpu().numpy()
            scores.extend(acceptable_probs)
    
    return scores


def load_unieval():
    """
    Load UniEval model for text quality evaluation.
    Returns coherence scores between 0-1 (higher = more coherent).
    """
    if not UNIEVAL_AVAILABLE:
        raise RuntimeError("UniEval is not available. Please clone the repository.")
    
    print("Loading UniEval model...")
    # Use 'dialogue' task for internal coherence evaluation (reference-free)
    # DialogEvaluator's coherence dimension evaluates text's internal coherence
    evaluator = get_evaluator('dialogue')
    print("UniEval model loaded successfully")
    return evaluator


def evaluate_unieval_batch(texts, evaluator, batch_size=8):
    """
    Evaluate coherence using UniEval DialogEvaluator (reference-free).
    Returns scores between 0-1 (higher = more coherent).
    """
    scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="UniEval"):
        batch_texts = texts[i:i+batch_size]
        
        # DialogEvaluator format: source (previous context), system_output (response), context (knowledge)
        # For standalone text coherence evaluation (no conversation context):
        # - source: empty (no previous dialogue turn)
        # - system_output: the text to evaluate
        # - context: empty (no external knowledge base)
        data = [{"source": "", "system_output": text, "context": ""} for text in batch_texts]
        
        try:
            # Evaluate only coherence dimension (internal coherence of the text)
            batch_scores = evaluator.evaluate(data, dims=['coherence'], overall=False, print_result=False)
            # Extract coherence scores
            coherence_scores = [result['coherence'] for result in batch_scores]
            scores.extend(coherence_scores)
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            # Add default scores for failed batch
            scores.extend([0.0] * len(batch_texts))
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate coherence using UniEval and DeBERTa-CoLA')
    parser.add_argument('--base-path', type=str, 
                    default="sample_results/openwebtext-split/topk_all_nucleus_all/standard/local_search_language/['toxicity', 'measure_combined_heuristic', 'perplexity']_lb[-100.0, -100.0, -100.0]_ub[0.5, 0.0, 0.0]/rank_1/top_p/0.0/local_5_candidates_numtheta_8_length100/seed_1/molecules",
                    help='Path to directory containing text files to evaluate')
    parser.add_argument('--model', type=str, choices=['unieval', 'deberta', 'both'], default='both',
                        help='Which model(s) to use for evaluation')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--output-name', type=str, default='coherence_model_scores.json',
                        help='Output JSON filename')
    args = parser.parse_args()

    # Check if UniEval is requested but not available
    if args.model in ['unieval', 'both'] and not UNIEVAL_AVAILABLE:
        print("ERROR: UniEval is not available but was requested.")
        print("Please clone UniEval: git clone https://github.com/maszhongming/UniEval.git")
        if args.model == 'unieval':
            exit(1)
        else:
            print("Falling back to DeBERTa only...")
            args.model = 'deberta'

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

    print(f"Loaded {len(sentences)} sentences")

    # Initialize results
    results = []
    for file_name, sentence in zip(file_names, sentences):
        results.append({
            "file": file_name,
            "sentence": sentence
        })

    # Evaluate with DeBERTa-CoLA
    if args.model in ['deberta', 'both']:
        print("\n" + "="*80)
        print("Evaluating with DeBERTa-v3-Large (CoLA)")
        print("="*80)
        deberta_model, deberta_tokenizer, device = load_deberta_cola()
        cola_scores = evaluate_cola_batch(
            sentences, 
            deberta_model, 
            deberta_tokenizer, 
            device,
            batch_size=args.batch_size
        )
        
        # Add to results
        for i, score in enumerate(cola_scores):
            results[i]['cola_score'] = float(score)
        
        # Print statistics
        print(f"\nCoLA Statistics:")
        print(f"  Mean: {np.mean(cola_scores):.4f}")
        print(f"  Std: {np.std(cola_scores):.4f}")
        print(f"  Min: {np.min(cola_scores):.4f}")
        print(f"  Max: {np.max(cola_scores):.4f}")
        
        # Clean up
        del deberta_model, deberta_tokenizer
        torch.cuda.empty_cache()

    # Evaluate with UniEval
    if args.model in ['unieval', 'both']:
        print("\n" + "="*80)
        print("Evaluating with UniEval (Coherence)")
        print("="*80)
        unieval = load_unieval()
        coherence_scores = evaluate_unieval_batch(
            sentences,
            unieval,
            batch_size=args.batch_size
        )
        
        # Add to results
        for i, score in enumerate(coherence_scores):
            results[i]['unieval_coherence'] = float(score)
        
        # Print statistics
        print(f"\nUniEval Coherence Statistics:")
        print(f"  Mean: {np.mean(coherence_scores):.4f}")
        print(f"  Std: {np.std(coherence_scores):.4f}")
        print(f"  Min: {np.min(coherence_scores):.4f}")
        print(f"  Max: {np.max(coherence_scores):.4f}")

    # Save results
    output_dir = os.path.join(base_path, "..")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Print combined statistics if both models were used
    if args.model == 'both' and len(results) > 0:
        print("\nCombined Statistics:")
        print(f"  Total samples: {len(results)}")
        
        # Check correlation between two metrics
        cola_vals = [r['cola_score'] for r in results]
        unieval_vals = [r['unieval_coherence'] for r in results]
        correlation = np.corrcoef(cola_vals, unieval_vals)[0, 1]
        print(f"  Correlation (CoLA vs UniEval): {correlation:.4f}")
        
        # Count high-quality samples (both metrics > 0.7)
        high_quality = sum(1 for r in results 
                          if r.get('cola_score', 0) > 0.7 and r.get('unieval_coherence', 0) > 0.7)
        print(f"  High quality samples (both > 0.7): {high_quality}/{len(results)} ({100*high_quality/len(results):.1f}%)")


if __name__ == "__main__":
    main()
