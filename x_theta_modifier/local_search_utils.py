import random
import time  # Import time module for timing functionality
import math  # Import math module for infinity
# Add RDKit SA_Score to path for sascorer import
from properties.peptides import *
from properties.molecules import *
from properties.trna import *
from properties.property_util import *

# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("yairschiff/qm9-tokenizer", trust_remote_code=True)

special_tokens = ['<bos>', '<eos>', '<mask>', '<pad>', '<unk>']

def local_search_qm9(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens):
    # Initialize best molecule
    tokens = fix_basic_syntax_tokens(tokens[:])
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"

    for i, token in enumerate(tokens):
        if token in atom_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, atom_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'atom substitution')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, atom_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'atom insertion')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def local_search_grampa(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens):
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"

    for i, token in enumerate(tokens):
        if token in aa_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, aa_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'amino acid substitution')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, aa_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'amino acid insertion')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found


    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def local_search_trna(tokens, distance_to_bounds, property_calcs, num_local_search_iters, max_num_local_search_iters, max_candidate_tokens):
    best_tokens = tokens[:]
    best_sequence = ''.join(token for token in best_tokens if token not in special_tokens)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    is_found = all(distance == 0 for distance in best_distances)
    operator = "Keep"
    if is_found:
        return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
    

    for i, token in enumerate(tokens):
        if token in nucleotide_candidates:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, nucleotide_candidates_with_pad, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'nucleotide substitution')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found
                
        elif token in special_tokens and (i == 0 or tokens[i - 1] not in special_tokens):
        # elif token in special_tokens and tokens[i - 1] not in special_tokens:
            best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters = substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, nucleotide_candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, 'nucleotide insertion')
            if is_found or is_max_num_local_search_iters:
                return best_tokens, best_prop_values, num_local_search_iters, operator, is_found


    return best_tokens, best_prop_values, num_local_search_iters, operator, is_found

def substitution_operator(best_tokens, best_prop_values, best_distances, tokens, i, token, num_local_search_iters, operator, candidates, property_calcs, distance_to_bounds, max_num_local_search_iters, max_candidate_tokens, operator_type):
    is_max_num_local_search_iters = False
    is_found = False
    num_candidate_tokens = 0
    num_local_search_iters += 1
    random.shuffle(candidates)

    for replacement in candidates:
        if token == replacement: # Skip replacing a token with itself
            continue
        num_candidate_tokens += 1
        # Replace the token with the candidate
        new_tokens = tokens[:]
        new_tokens[i] = replacement
        new_sequence = ''.join(token for token in new_tokens if token not in special_tokens)
        new_prop_values = [calc(new_sequence) for calc in property_calcs]
        new_distances = [dist(value) for dist, value in zip(distance_to_bounds, new_prop_values)]

        # Check if the new molecule is better
        if compare_hierarchical(new_distances, best_distances) < 0:
            operator = operator_type
            best_tokens = new_tokens
            best_prop_values = new_prop_values
            best_distances = new_distances
            if all(distance == 0 for distance in best_distances):
                is_found = True
                return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters
        # Break if maximum iterations are reached
        if num_local_search_iters >= max_num_local_search_iters:
            is_max_num_local_search_iters = True
            return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters
        if num_candidate_tokens >= max_candidate_tokens:
            break # Only try `max_candidate_tokens` replacements per position

    return best_tokens, best_prop_values, best_distances, num_local_search_iters, operator, is_found, is_max_num_local_search_iters

def local_search_language(best_token_ids, x_theta_probs, distance_to_bounds, property_calcs, tokenizer, top_k_values_for_local_search, device):
    """
    Local search for language data using top-k values from x_theta probabilities.
    For each position, only try the top-k most probable tokens.
    
    Args:
        best_token_ids: Current best sequence (1D tensor of token IDs)
        x_theta_probs: Probability distribution over vocab for each position (seq_len x vocab_size)
        distance_to_bounds: List of distance functions
        property_calcs: List of property calculation functions
        tokenizer: Tokenizer for decoding sequences
        top_k_values_for_local_search: Number of top-k values to try per position
        device: Device to use
    
    Returns:
        best_token_ids: Best sequence found (1D tensor)
    """
    seq_len = best_token_ids.shape[0]
    
    # Calculate initial properties
    best_sequence = tokenizer.decode(best_token_ids, skip_special_tokens=True)
    best_prop_values = [calc(best_sequence) for calc in property_calcs]
    best_distances = [dist(value) for dist, value in zip(distance_to_bounds, best_prop_values)]
    
    # Get top-k token indices for each position from x_theta
    # x_theta_probs shape: (seq_len, vocab_size)
    topk_values, topk_indices = torch.topk(x_theta_probs, k=top_k_values_for_local_search, dim=-1)
    # topk_indices shape: (seq_len, top_k_values_for_local_search)
    
    # Iterate through each rank (1st best, 2nd best, ..., k-th best)
    for k_rank in range(top_k_values_for_local_search):
        # Iterate through each position in sequence
        for pos in range(seq_len):
            # Get the k-th best token for this position
            candidate_token_id = topk_indices[pos, k_rank].item()
            
            # Skip if it's the same as current token
            if candidate_token_id == best_token_ids[pos].item():
                continue
            
            # Create neighbor sequence by changing only this position
            neighbor_token_ids = best_token_ids.clone()
            neighbor_token_ids[pos] = candidate_token_id
            
            # Evaluate neighbor
            neighbor_sequence = tokenizer.decode(neighbor_token_ids, skip_special_tokens=True)
            neighbor_prop_values = [calc(neighbor_sequence) for calc in property_calcs]
            neighbor_distances = [dist(value) for dist, value in zip(distance_to_bounds, neighbor_prop_values)]
            
            # Check if neighbor is better
            if compare_hierarchical(neighbor_distances, best_distances) < 0:
                best_token_ids = neighbor_token_ids
                best_prop_values = neighbor_prop_values
                best_distances = neighbor_distances
    
    return best_token_ids


def local_search_on_best_tokens(best_tokens, tokenizer, num_local_searches, distance_to_bounds, property_calcs, local_search, device, max_candidate_tokens, x_theta=None, top_k_values_for_local_search=None):
    """
    Apply local search on best tokens.
    
    Args:
        best_tokens: Best sequences (batch_size x seq_len)
        tokenizer: Tokenizer
        num_local_searches: Number of local search iterations
        distance_to_bounds: Distance functions
        property_calcs: Property calculation functions
        local_search: Local search function
        device: Device
        max_candidate_tokens: Max candidate tokens to try
        x_theta: Optional x_theta probabilities for language local search (batch_size x seq_len x vocab_size)
        top_k_values_for_local_search: Optional top-k value for language local search
    
    Returns:
        Modified best tokens
    """
    # Check if we should use language local search
    use_language_search = (x_theta is not None and top_k_values_for_local_search is not None)
    
    # Iterate through each best_token in best_tokens
    modified_tokens = []
    for best_token_index, best_token in enumerate(best_tokens):
        # Move best_token to CPU for operations
        best_token_cpu = best_token.cpu()
        
        if use_language_search:
            # Use new language-specific local search with top-k values from x_theta
            x_theta_probs_for_sequence = x_theta[best_token_index]  # (seq_len, vocab_size)
            modified_token_ids = local_search_language(
                best_token_ids=best_token_cpu,
                x_theta_probs=x_theta_probs_for_sequence.cpu(),
                distance_to_bounds=distance_to_bounds,
                property_calcs=property_calcs,
                tokenizer=tokenizer,
                top_k_values_for_local_search=top_k_values_for_local_search,
                device=device
            )
            modified_tokens.append(modified_token_ids.tolist())
        else:
            # Use original local search for molecules/peptides/etc
            # Convert token IDs to tokens directly to preserve length
            sequence_string = tokenizer.decode(best_token_cpu, skip_special_tokens=False)
            modified_token = tokenizer.tokenize(sequence_string)

            max_num_local_search_iters = 1e9
            index_local_search = 0
            num_local_search_iters = 0
            is_found = False

            while index_local_search < num_local_searches and num_local_search_iters < max_num_local_search_iters and not is_found:
                new_best_tokens, best_prop_values, num_local_search_iters, operator, is_found = local_search(
                    tokens=modified_token,
                    distance_to_bounds=distance_to_bounds,
                    property_calcs=property_calcs,
                    num_local_search_iters=num_local_search_iters,
                    max_num_local_search_iters=max_num_local_search_iters,
                    max_candidate_tokens=max_candidate_tokens
                )
                index_local_search += 1
                if new_best_tokens == modified_token:
                    break
                else:
                    modified_token = new_best_tokens[:]

            modified_token_ids = tokenizer.convert_tokens_to_ids(modified_token)
            modified_tokens.append(modified_token_ids)

    # Stack into a tensor with proper device
    best_tokens = torch.tensor(modified_tokens, dtype=torch.long, device=device)
    return best_tokens

if __name__ == "__main__":
    smiles = "C1=CC=CC=)C1O"
    import ipdb; ipdb.set_trace()