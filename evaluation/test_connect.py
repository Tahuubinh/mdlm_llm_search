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
                    default="sample_results/openwebtext-split/topk_all_nucleus_all/standard/local_search_language/['toxicity', 'measure_combined_heuristic', 'perplexity']_lb[-100.0, -100.0, -100.0]_ub[0.5, 0.0, 0.0]/rank_1/top_p/0.0/local_5_candidates_numtheta_8_length100/seed_0/molecules",
                    help='Path to directory containing text files to evaluate')
parser.add_argument('--batch-size', type=int, default=30, help='Number of queries before rate limiting')
parser.add_argument('--sleep-time', type=int, default=61, help='Sleep time in seconds after each batch')
args = parser.parse_args()

client = genai.Client(api_key=args.api_key)



response = client.models.generate_content(
    model="gemma-3-27b-it", 
    contents="Hello, are you ready?"
)
print(response.text)