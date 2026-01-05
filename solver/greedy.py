import numpy as np

def greedy_search(A: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Greedy construction heuristic for Set Packing Problem
    
    Selects sets iteratively based on utility (weight/occurrence ratio),
    removing conflicting sets after each selection.
    """
    n_sets = c.shape[0]
    
    # Initialize solution and candidate set
    x0 = np.zeros(n_sets, dtype=int)
    c0 = np.ones(n_sets, dtype=bool)  # All sets initially selectable
    
    # Count occurrences (items per set)
    occurrences = np.count_nonzero(A, axis=0)  # More concise
    
    # Avoid division by zero (defensive programming)
    occurrences = np.maximum(occurrences, 1e-10)
    
    # Greedy construction
    while np.any(c0):
        # Calculate utility for selectable candidates
        utility = np.where(c0, c / occurrences, 0.0)
        
        # Select set with highest utility
        index_max = np.argmax(utility)
        
        # Add to solution
        x0[index_max] = 1
        c0[index_max] = False
        
        # Eliminate conflicts
        saturated_rows = np.where(A[:, index_max] == 1)[0]
        
        for row in saturated_rows:
            conflicting_sets = np.where(A[row, :] == 1)[0]
            c0[conflicting_sets] = False  # Vectorized elimination
        
        c0[index_max] = False  # Ensure selected set stays removed
    
    return x0


def is_admissible(A: np.ndarray, x0: np.ndarray) -> bool:
    """Check if solution is feasible (each item used at most once)"""
    return np.all(A @ x0 <= 1)