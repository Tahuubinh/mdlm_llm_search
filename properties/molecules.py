import torch
from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit.Chem import RDConfig
from rdkit import RDLogger
import sys
import os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # type: ignore
RDLogger.DisableLog('rdApp.*')

atom_candidates = [
    'C','O','N','F','c','n','o',
    '[C-]','[CH-]','[N+]','[N-]','[NH+]','[NH2+]','[NH3+]','[O-]',
    '[c-]','[cH-]','[n-]','[nH+]','[nH]'
]

branch_candidates = [
    ['(', 'C', ')'],  # Methyl / carbon đơn
    ['(', 'O', ')'],  # Hydroxyl
    ['(', 'N', ')'],  # Amine
    ['(', 'F', ')'],  # Fluoro
    ['(', 'c', ')'],  # Aromatic carbon
    ['(', 'n', ')'],  # Aromatic nitrogen
    ['(', 'o', ')'],  # Aromatic oxygen
    ['(', '[C-]', ')'],
    ['(', '[CH-]', ')'],
    ['(', '[N+]', ')'],
    ['(', '[N-]', ')'],
    ['(', '[NH+]', ')'],
    ['(', '[NH2+]', ')'],
    ['(', '[NH3+]', ')'],
    ['(', '[O-]', ')'],
    ['(', '[c-]', ')'],
    ['(', '[cH-]', ')'],
    ['(', '[n-]', ')'],
    ['(', '[nH+]', ')'],
    ['(', '[nH]', ')']
]

ring_candidates = [
    ['C','1','C','C','1'],          # 3-member carbon
    ['C','1','C','C','C','1'],      # 4-member carbon
    ['C','1','C','C','C','C','1'],  # 5-member carbon
    ['O','1','C','O','1'],          # 3-member O
    ['O','1','C','C','O','1'],      # 4-member O
    ['N','1','C','N','1'],          # 3-member N
    ['N','1','C','C','N','1'],      # 4-member N
    ['c','1','c','c','1'],          # 3-member aromatic C
    ['c','1','c','c','c','1'],      # 4-member aromatic C
    ['n','1','c','n','1']           # 3-member aromatic N
]

atom_candidates_with_pad = atom_candidates + ['<pad>']

def fix_basic_syntax(smiles: str) -> str:
    """
    Fix basic syntax errors in a SMILES string by balancing parentheses.
    Removes unmatched '(' or ')' from the SMILES.
    """
    result = []
    open_count = 0

    for ch in smiles:
        if ch == '(':
            open_count += 1
            result.append(ch)
        elif ch == ')':
            if open_count > 0:
                open_count -= 1
                result.append(ch)
        else:
            result.append(ch)

    final = []
    for ch in reversed(result):
        if ch == '(' and open_count > 0:
            open_count -= 1
        else:
            final.append(ch)
    smiles = ''.join(reversed(final))

    for d in "123456789":
        count = smiles.count(d)
        if count % 2 != 0:
            idx = smiles.rfind(d)
            if idx != -1:
                smiles = smiles[:idx] + smiles[idx+1:]

    return smiles

def fix_basic_syntax_tokens(tokens):
    # Replace excess ')' with <pad> from left to right
    open_count = 0
    for i, token in enumerate(tokens):
        if token == '(':
            open_count += 1
        elif token == ')':
            if open_count > 0:
                open_count -= 1
            else:
                tokens[i] = '<pad>'

    # Replace excess '(' with <pad> from right to left
    close_count = 0
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == ')':
            close_count += 1
        elif tokens[i] == '(':
            if close_count > 0:
                close_count -= 1
            else:
                tokens[i] = '<pad>'

    # Balance ring numbers 1–9
    ring_counts = {str(i): 0 for i in range(1, 10)}
    for token in tokens:
        if token in ring_counts:
            ring_counts[token] += 1

    for i, token in enumerate(tokens):
        if token in ring_counts and ring_counts[token] % 2 != 0:
            tokens[i] = '<pad>'
            ring_counts[token] -= 1

    return tokens

# Function to calculate logP
def calc_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Crippen.MolLogP(mol)

# Function to calculate QED (Quantitative Estimate of Drug-likeness)
def calc_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return QED.qed(mol)
    except Chem.rdchem.KekulizeException:
        print(f"Can't kekulize molecule: {smiles}")
        return None
    except Exception as e:
        print(f"Error calculating QED for molecule {smiles}: {e}")
        return None

def calc_sa(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)

def calc_validity(smiles):
    if not smiles:
        return -1
    mol = Chem.MolFromSmiles(smiles)
    return 1 if mol else 0

# Function to calculate QED (Quantitative Estimate of Drug-likeness)
def calc_qed_parallel(smiles_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, smiles in enumerate(smiles_list):
        # Clean the SMILES string
        cleaned_smiles = smiles.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<unk>', '').strip()
        cleaned_smiles = fix_basic_syntax(cleaned_smiles)
        if cleaned_smiles:
            # Try to create molecule from SMILES
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol:
                try:
                    rewards[i] = QED.qed(mol)
                except:
                    rewards[i] = float('nan')
            else:
                rewards[i] = float('nan')
        else:
            rewards[i] = float('nan')

    return rewards

def calc_sa_parallel(smiles_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, smiles in enumerate(smiles_list):
        # Clean the SMILES string
        cleaned_smiles = smiles.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<unk>', '').strip()
        cleaned_smiles = fix_basic_syntax(cleaned_smiles)
        if cleaned_smiles:
            # Try to create molecule from SMILES
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol:
                try:
                    rewards[i] = sascorer.calculateScore(mol)
                except:
                    rewards[i] = float('nan')
            else:
                rewards[i] = float('nan')
        else:
            rewards[i] = float('nan')

    return rewards

def calc_validity_parallel(smiles_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, smiles in enumerate(smiles_list):
        # Clean the SMILES string
        cleaned_smiles = smiles.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<unk>', '').strip()
        if cleaned_smiles:
            # Try to create molecule from SMILES
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol:
                rewards[i] = 1
            else:
                rewards[i] = 0 
        else:
            rewards[i] = 0.2 

    return rewards

def modify_until_valid(smiles, max_num_difference, tokenizer):
    def generate_variants(tokens, num_differences):
        if num_differences == 0:
            yield tokens
            return
        for i in range(len(tokens)):
            original_token = tokens[i]
            for replacement in tokenizer.vocab.keys():  # Iterate through tokenizer's vocabulary
                if replacement != original_token:
                    new_tokens = tokens[:]
                    new_tokens[i] = replacement
                    yield from generate_variants(new_tokens, num_differences - 1)

    tokens = tokenizer.tokenize(smiles)
    for num_differences in range(1, max_num_difference + 1):
        print(f"Trying {num_differences} token differences...")
        for variant_tokens in generate_variants(tokens, num_differences):
            variant_smiles = ''.join(str(token) for token in variant_tokens)
            if Chem.MolFromSmiles(variant_smiles) is not None:
                return variant_smiles
    return smiles

def modify_toward_valid(smiles, max_num_difference, tokenizer):
    def generate_variants(tokens, num_differences):
        if num_differences == 0:
            yield tokens
            return
        for i in range(len(tokens)):
            original_token = tokens[i]
            for replacement in tokenizer.vocab.keys():  # Iterate through tokenizer's vocabulary
                if replacement != original_token:
                    new_tokens = tokens[:]
                    new_tokens[i] = replacement
                    yield from generate_variants(new_tokens, num_differences - 1)

    tokens = tokenizer.tokenize(smiles)
    for num_differences in range(1, max_num_difference + 1):
        print(f"Trying {num_differences} token differences...")
        for variant_tokens in generate_variants(tokens, num_differences):
            variant_smiles = ''.join(str(token) for token in variant_tokens)
            if Chem.MolFromSmiles(variant_smiles) is not None:
                return variant_smiles
    return smiles