import time
import os
import sys
import json
import re
import argparse
# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
try:
    from google import genai
except ImportError:
    print("Error: Please install google-genai package")
    print("Run: pip install google-genai")
    exit(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate coherence of generated sentences using Gemma model')
parser.add_argument('--api-key', type=str, required=True, help='Google GenAI API key')
parser.add_argument('--base-path', type=str, 
                    default="sample_results/openwebtext-split/topk_all_nucleus_all/standard/local_search_language/['toxicity', 'measure_combined_heuristic', 'perplexity']_lb[-100.0, -100.0, -100.0]_ub[0.5, 0.0, 0.0]/rank_1/top_p/0.0/local_5_candidates_numtheta_8_length100/seed_1/molecules",
                    help='Path to directory containing text files to evaluate')
parser.add_argument('--batch-size', type=int, default=30, help='Number of queries before rate limiting')
parser.add_argument('--sleep-time', type=int, default=61, help='Sleep time in seconds after each batch')
args = parser.parse_args()

# Initialize the client
client = genai.Client(api_key=args.api_key)
scores = []
BATCH_SIZE = args.batch_size
SLEEP_TIME = args.sleep_time

# Load text files from the specified directory
base_path = args.base_path

# Get all .txt files in the directory
# Note: Using os.listdir instead of glob because path contains special characters like []
if not os.path.exists(base_path):
    print(f"ERROR: Path does not exist: {base_path}")
    exit(1)

all_files = os.listdir(base_path)
txt_files = sorted([os.path.join(base_path, f) for f in all_files if f.endswith('.txt')])
print(f"Found {len(txt_files)} text files in {base_path}")

# Read all sentences from the files
decoded_sentences = []
file_names = []
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        decoded_sentences.append(content)
        file_names.append(os.path.basename(txt_file))

total_sentences = len(decoded_sentences)
print(f"Starting evaluation of {total_sentences} sentences.")

# Define output file path early - save to parent directory
output_dir = os.path.join(base_path, "..")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "coherence_scores.json")
temp_file = output_file + ".tmp"

start_time = time.time()
for i, sentence in enumerate(decoded_sentences):
    prompt = (
        f"You are a language expert evaluating the fluency and coherence of the following AI-generated sentence. "
        f"Fluency means grammatical correctness and natural language usage. Coherence means logical clarity and connectedness. "
        f"Give a single integer score between 0 (poor) and 100 (excellent), with no explanation or comments.\n\n"
        f"Sentence: {sentence}\nScore:"
    )
    try:
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt
        )
        raw_text = response.text.strip() if response and response.text else "0"
        
        # Use regex to extract first integer (handles cases like "Score: 85" or "85/100")
        match = re.search(r'\d+', raw_text)
        score_text = match.group() if match else "0"
        
        # Log raw response for debugging (optional, can be removed in production)
        print(f"  [DEBUG] Raw response for sentence {i+1}: '{raw_text}' -> Score: {score_text}")
        
    except Exception as e:
        print(f"Error processing sentence {i+1}: {e}")
        score_text = "NA"  # Assign a default score if an error occurs
    
    scores.append(score_text)
    
    # Print progress details
    current_count = i + 1
    elapsed = time.time() - start_time
    progress_frac = current_count / total_sentences
    print(f"Processed sentence {current_count}/{total_sentences} ({progress_frac*100:.1f}% done). Elapsed: {elapsed:.1f}s")
    
    # --- INCREMENTAL BACKUP (save after each batch to prevent data loss) ---
    if current_count % BATCH_SIZE == 0:
        temp_results = []
        for j in range(len(scores)):
            temp_results.append({
                "file": file_names[j],
                "sentence": decoded_sentences[j],
                "coherence_score": scores[j]
            })
        with open(temp_file, 'w', encoding='utf-8') as f_tmp:
            json.dump(temp_results, f_tmp, indent=2, ensure_ascii=False)
        print(f"  → Saved temporary backup to {temp_file}")
    
    # Rate-limiting
    if current_count % BATCH_SIZE == 0 and current_count < total_sentences:
        print(f"Reached {current_count} queries; sleeping for {SLEEP_TIME} seconds...")
        time.sleep(SLEEP_TIME)

print("------------------------------------------------------------")
print(f"Done! Collected {len(scores)} scores for all sentences.")

# Save final results to JSON file
results = []
for i, (file_name, sentence, score) in enumerate(zip(file_names, decoded_sentences, scores)):
    results.append({
        "file": file_name,
        "sentence": sentence,
        "coherence_score": score
    })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Results saved to: {output_file}")

# Remove temporary file if completed successfully
if os.path.exists(temp_file):
    os.remove(temp_file)
    print(f"Removed temporary file: {temp_file}")

# Calculate statistics for numeric scores
numeric_scores = []
for score in scores:
    try:
        # Since we used regex extraction, score_text should be pure number or "NA"
        if score != "NA":
            numeric_scores.append(float(score))
    except:
        pass

if numeric_scores:
    avg_score = sum(numeric_scores) / len(numeric_scores)
    min_score = min(numeric_scores)
    max_score = max(numeric_scores)
    print(f"\nStatistics:")
    print(f"  Average score: {avg_score:.2f}")
    print(f"  Min score: {min_score:.2f}")
    print(f"  Max score: {max_score:.2f}")
    print(f"  Valid scores: {len(numeric_scores)}/{len(scores)}")