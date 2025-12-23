"""Properties and metrics for tRNA sequences."""
import RNA
import torch

# Standard nucleotides for tRNA
nucleotide_candidates = ["A", "C", "G", "U", "N"]
nucleotide_candidates_with_pad = nucleotide_candidates + ['<pad>']

def calc_trna_length(sequence):
    return len(sequence)

def is_watson_crick(bp1, bp2):
    return (bp1, bp2) in [('A','U'), ('U','A'), ('G','C'), ('C','G')]

def is_wobble(bp1, bp2):
    return (bp1, bp2) in [('G','U'), ('U','G')]

def count_acceptor_stem(sequence, min_bp=7, max_bp=9):
    """
    Check the Acceptor Stem.
    Constraint: 7-9 bp (acceptor stem may contain non-Watson-Crick base pairs).
    NOTE: Sequence from GtRNAdb usually has the format: 1...72 + 73(Discriminator).
    No CCA tail.
    Return: int - number of valid base pairs (Watson-Crick or wobble)
    """
    count = 0
    # Check up to max_bp pairs
    for i in range(max_bp):
        if i >= len(sequence) // 2:  # Prevent overflow if sequence is too short
            break
        base_5 = sequence[i]
        base_3 = sequence[-(i + 2)]  # Skip Discriminator
        
        if is_watson_crick(base_5, base_3) or is_wobble(base_5, base_3):
            count += 1
            
    return count

def get_t_loop_score(sequence):
    """
    Calculate the score for the TΨC loop motif.
    The canonical motif is 5'-TΨCGA-3' (GUUCGA in normalized RNA).
    Input: Sequence (String)
    Output: Highest score found (Int from 0 to 6)
            - 6: Perfect Match (Valid)
            - <6: Not valid
    """
    
    # Common TΨC loop motif variants
    # In databases, Ψ (pseudouridine) and T (ribothymidine) are normalized to U
    targets = [
        "GUUCGA",  # Most standard: 5'-TΨCGA-3'
        "GUUCAA",  # Very common variant in mammals (A instead of G at position 5)
        "GUUCGG",  # G at position 5 (purine variant)
        "GUUCAG",  # Less common variant
    ]
    
    max_score = 0

    for i in range(len(sequence) - 6 + 1):
        window = sequence[i : i+6]

        # Compare the current window with all target motifs
        for target in targets:
            score = sum(1 for a, b in zip(window, target) if a == b)
            if score > max_score:
                max_score = score
                
        # If a score of 6 (Perfect) is found, stop immediately for efficiency
        if max_score == 6:
            return 6
            
    return max_score

def fold_structure(sequence):
    """Fold tRNA sequence into secondary structure."""
    sequence = sequence.upper().replace("T", "U")
    fc = RNA.fold_compound(sequence)
    struct, mfe = fc.mfe()
    return struct, mfe

def find_stems_ptable(pt, min_bp=3):
    """
    Find stems from the pair table. Returns a list of stems with the format:
    [(i_start, j_start, length), ...] 
    i_start: start position 5' (0-indexed)
    j_start: start position 3' (0-indexed)
    length: number of base pairs
    """
    n = pt[0]  # pt[0] contains the length
    visited = [False] * (n + 1)
    stems = []
    
    for i in range(1, n + 1):
        j = pt[i]
        if j > i and not visited[i]:  # Only consider pairs i < j
            # Count consecutive base pairs
            length = 0
            i_curr, j_curr = i, j
            while i_curr <= n and j_curr >= 1 and pt[i_curr] == j_curr:
                visited[i_curr] = True
                visited[j_curr] = True
                length += 1
                i_curr += 1
                j_curr -= 1
            
            if length >= min_bp:
                stems.append((i - 1, j - 1, length))  # Convert to 0-indexed
    
    return stems

def identify_stems_by_position(stems, sequence_length):
    """
    Identify stems based on RELATIVE POSITION (%) in the tRNA structure.
    This ensures the code works well with tRNAs of different lengths (70-76 nt).

    tRNA cloverleaf structure (relative positions):
    - Acceptor stem: starts ~0-5%, ends ~80-100%
    - D-stem: starts ~8-22%, ends ~23-40%  
    - Anticodon stem: starts ~28-48%, ends ~44-65%
    - T-stem: starts ~60-78%, ends ~75-95%

    Return: dict with keys: 'acceptor', 'd_stem', 'anticodon', 't_stem'
    """
    if len(stems) < 3:
        return None
    
    acceptor = None
    d_stem = None
    anticodon = None
    t_stem = None
    
    L = sequence_length
    
    for i_start, j_start, length in stems:
        i_pct = i_start / L * 100
        j_pct = j_start / L * 100
        
        # Acceptor: starts 0-7%, ends 78-100%
        if i_pct <= 7 and j_pct >= 78:
            acceptor = (i_start, j_start, length)
        
        # D-stem: starts 8-22%, ends 23-42%
        elif 8 <= i_pct <= 22 and 23 <= j_pct <= 42:
            d_stem = (i_start, j_start, length)
        
        # Anticodon: starts 28-48%, ends 44-66%
        elif 28 <= i_pct <= 48 and 44 <= j_pct <= 66:
            anticodon = (i_start, j_start, length)
        
        # T-stem: starts 60-78%, ends 75-96%
        elif 60 <= i_pct <= 78 and 75 <= j_pct <= 96:
            t_stem = (i_start, j_start, length)
    
    return {
        'acceptor': acceptor,
        'd_stem': d_stem,
        'anticodon': anticodon,
        't_stem': t_stem
    }

def get_stem_topology(sequence):
    """
    Fold the sequence only once and extract the topology of all stems.
    Return: dict with keys: 'd_stem', 'anticodon', 't_stem' (each value is a tuple (i, j, length) or None)
    """
    struct, _ = fold_structure(sequence)
    pt = RNA.ptable(struct)
    stems = find_stems_ptable(pt, min_bp=3)
    
    # Identify stems based on POSITION (not index)
    topology = identify_stems_by_position(stems, len(sequence))
    
    if not topology:
        return {'d_stem': None, 'anticodon': None, 't_stem': None}
    
    return {
        'd_stem': topology['d_stem'],
        'anticodon': topology['anticodon'],
        't_stem': topology['t_stem']
    }

def check_fold_structure_constraints(sequence):
    """
    Check the constraints on the secondary structure of tRNA.
    IMPORTANT: Fold the sequence ONLY ONCE to optimize performance.

    Return: number of constraints satisfied (0-3)
    - D-stem: 4-6 bp
    - Anticodon stem: 5 bp  
    - Variable loop: 3-21 bases
    """
    # Fold ONLY ONCE
    topology = get_stem_topology(sequence)
    
    # D-stem constraint (4-6 bp)
    d_ok = False
    if topology['d_stem']:
        d_len = topology['d_stem'][2]
        d_ok = 4 <= d_len <= 6
    
    # Anticodon stem constraint (5 bp)
    ac_ok = False
    if topology['anticodon']:
        ac_len = topology['anticodon'][2]
        ac_ok = ac_len == 5
    
    # Variable loop constraint (3-21 bases)
    v_ok = False
    if topology['anticodon'] and topology['t_stem']:
        ac_i, ac_j, ac_len = topology['anticodon']
        t_i, t_j, t_len = topology['t_stem']
        v_loop_size = t_i - ac_j - 1
        v_ok = 3 <= v_loop_size <= 21
    
    return d_ok + ac_ok + v_ok

def calc_trna_length_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = calc_trna_length(sequence)
    return rewards

def count_acceptor_stem_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = count_acceptor_stem(sequence)
    return rewards

def get_t_loop_score_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = get_t_loop_score(sequence)
    return rewards

def check_fold_structure_constraints_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = check_fold_structure_constraints(sequence)
    return rewards

def calculate_mfe(sequence):
    structure, mfe = RNA.fold(sequence)
    return mfe