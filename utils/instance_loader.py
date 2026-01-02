# utils/instance_loader.py
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_instance(filename: str) -> Dict:
    #filename:str and -> Dict are type hints that help     
    
    """
    Load SPP instance from OR-Library format .dat file
    
    Format:
        Line 1: n_items n_sets
        Line 2: weights for each set
        Then for each set:
            - Number of elements in set
            - Elements in set (space-separated)
    
    Args:
        filename: Name of .dat file in data/ directory
    
    Returns:
        dict with:
            - n_items: number of items
            - n_sets: number of sets
            - weights: list of set weights
            - sets: list of lists (each inner list is a set)
    """
    filepath = Path(__file__).parent.parent / "data" / filename
  
    with open(filepath, 'r') as f:     
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    
    # Parse first line: nbr of constraints (m) & nbr of variables (n)
    m, n = map(int, lines[0].split())
    
    # Parse second line: coefficients of the objective function  (weights)
    weights = list(map(int, lines[1].split()))
    # Parse sets (variables) 
    sets = [] 
    line_idx = 2
    
    for i in range(m):
        # Number of elements in this set
        set_size = int(lines[line_idx])
        line_idx += 1
    
        # Elements in this set
        if set_size > 0:
            elements = list(map(int, lines[line_idx].split()))
            if line_idx < len(lines):
                line_idx += 1
        else:
            elements = []

        sets.append(elements)
    
    # Create coefficient vector c (weights)
    c = np.array(weights)
    
    # Create constraint matrix A
    # A[i,j] = 1 if item i is in set j, 0 otherwise
    # Rows = items (1 to n_items), Columns = sets (0 to n_sets-1)
    A = np.zeros((m, n), dtype=int)
    
    for i, set_elements in enumerate(sets):
        for j in set_elements:
            # Items are 1-indexed in the data, but array is 0-indexed
            A[i, j-1] = 1

    return {
        'n_constraints': n,
        'n_variables': m,
        'weights': weights,  # Keep as list for convenience
        'sets': sets,        # Keep original sets representation
        'c': c,              # Coefficient vector (numpy array)
        'A': A,              # Constraint matrix (numpy array)
        'filename': filename
    }
    

def list_instances() -> List[str]:
    """List all .dat files in data directory"""
    data_dir = Path(__file__).parent.parent / "data"
    return sorted([f.name for f in data_dir.glob("*.dat")])


def get_instance_info(instance: Dict) -> str:
    """Get human-readable info about an instance"""
    return f"{instance['n_constraints']} constraints, {instance['n_variables']} variables"