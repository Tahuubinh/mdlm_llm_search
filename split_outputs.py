import os
import re

# Input file path
input_file = "sample_results/openwebtext-split/topk_all_nucleus_all/standard/bon/['toxicity', 'perplexity']_lb[-100.0, -100.0]_ub[0.75, 0.0]/test/seed_1/threshold_75_outputs.txt"

# Output directory
output_dir = "sample_results/openwebtext-split/topk_all_nucleus_all/standard/bon/['toxicity', 'perplexity']_lb[-100.0, -100.0]_ub[0.75, 0.0]/test/seed_1/molecules"

# Prefix directory
prefix_dir = "data/toxicity/1000_samples"

# Option to remove prefix
remove_prefix = True

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the input file
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Process each line
for line in lines:
    # Match pattern: number followed by colon and space
    match = re.match(r'^(\d+):\s*(.*)$', line)
    
    if match:
        output_num = match.group(1)
        content = match.group(2)
        
        # Remove prefix if enabled
        if remove_prefix:
            prefix_file = os.path.join(prefix_dir, f"{output_num}.txt")
            if os.path.exists(prefix_file):
                with open(prefix_file, 'r', encoding='utf-8') as pf:
                    prefix = pf.read().strip()
                    # Remove prefix from content
                    if content.startswith(prefix):
                        content = content[len(prefix):]
                        print(f"Removed prefix for output {output_num}")
            else:
                print(f"Warning: Prefix file not found for output {output_num}")
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{output_num}.txt")
        
        # Write content to individual file
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(content)
        
        print(f"Created: {output_file}")

print(f"\nDone! Created {len(lines)} files in {output_dir}")
