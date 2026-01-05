
import numpy as np
from typing import Tuple

def is_admissible_neighbor(A: np.ndarray, x: np.ndarray, k: int) -> bool:
    """
    Test admissibility when setting variable k to 1
    
    Only tests constraints affected by setting x[k] = 1, since setting to 0
    never causes infeasibility.
    
    Args:
        A: Constraint matrix
        x: Current solution (potentially with x[k] = 1)
        k: Index of variable being set to 1
        
    Returns:
        True if solution remains feasible, False otherwise
    """
    # Find constraints where A[i,k] == 1 (affected by setting x[k] = 1)
    affected_rows = np.where(A[:, k] == 1)[0]
    
    # Check each affected constraint
    for row in affected_rows:
        if np.dot(A[row, :], x) > 1:
            return False
    
    return True


# ============================================================================
# K-P EXCHANGE NEIGHBORHOODS
# ============================================================================

def two_one_exchange(A: np.ndarray, x0: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    2-1 exchange: Remove 2 sets, add 1 set
    
    Explores neighborhood by:
    - Setting x[i] = 0 (remove first set)
    - Setting x[j] = 0 (remove second set)  
    - Setting x[k] = 1 (add new set)
    
    Returns first improving and feasible neighbor found.
    
    Args:
        A: Constraint matrix
        x0: Current solution
        c: Objective coefficients
        
    Returns:
        (improved_solution, found_improvement)
    """
    x_neighbor = x0.copy()
    z_current = np.dot(x0, c)
    n = len(x0)
    
    # Try removing first set (i)
    for i in range(n):
        if x0[i] != 1:
            continue
            
        x_neighbor[i] = 0  # Remove set i
        
        # Try removing second set (j)
        for j in range(n):
            if x0[j] != 1 or j == i:
                continue
                
            x_neighbor[j] = 0  # Remove set j
            
            # Try adding new set (k)
            for k in range(n):
                if x0[k] != 0:
                    continue
                    
                x_neighbor[k] = 1  # Add set k
                
                # Check if improving
                z_neighbor = np.dot(x_neighbor, c)
                if z_neighbor > z_current:
                    # Check if feasible
                    if is_admissible_neighbor(A, x_neighbor, k):
                        return x_neighbor, True
                
                x_neighbor[k] = 0  # Reset
            
            x_neighbor[j] = 1  # Reset
        
        x_neighbor[i] = 1  # Reset
    
    return x0, False


def one_one_exchange(A: np.ndarray, x0: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    1-1 exchange: Remove 1 set, add 1 set (swap)
    
    Explores neighborhood by:
    - Setting x[i] = 0 (remove set)
    - Setting x[j] = 1 (add set)
    
    Returns first improving and feasible neighbor found.
    
    Args:
        A: Constraint matrix
        x0: Current solution
        c: Objective coefficients
        
    Returns:
        (improved_solution, found_improvement)
    """
    x_neighbor = x0.copy()
    z_current = np.dot(x0, c)
    n = len(x0)
    
    # Try removing each set in solution
    for i in range(n):
        if x0[i] != 1:
            continue
            
        x_neighbor[i] = 0  # Remove set i
        
        # Try adding each set not in solution
        for j in range(n):
            if x0[j] != 0:
                continue
                
            x_neighbor[j] = 1  # Add set j
            
            # Check if improving
            z_neighbor = np.dot(x_neighbor, c)
            if z_neighbor > z_current:
                # Check if feasible
                if is_admissible_neighbor(A, x_neighbor, j):
                    return x_neighbor, True
            
            x_neighbor[j] = 0  # Reset
        
        x_neighbor[i] = 1  # Reset
    
    return x0, False


def zero_one_exchange(A: np.ndarray, x0: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    0-1 exchange: Add 1 set (no removal)
    
    Explores neighborhood by setting x[i] = 1 for sets not in solution.
    
    Returns first improving and feasible neighbor found.
    
    Args:
        A: Constraint matrix
        x0: Current solution
        c: Objective coefficients
        
    Returns:
        (improved_solution, found_improvement)
    """
    x_neighbor = x0.copy()
    z_current = np.dot(x0, c)
    n = len(x0)
    
    # Try adding each set not in solution
    for i in range(n):
        if x0[i] != 0:
            continue
            
        x_neighbor[i] = 1  # Add set i
        
        # Check if improving
        z_neighbor = np.dot(x_neighbor, c)
        if z_neighbor > z_current:
            # Check if feasible
            if is_admissible_neighbor(A, x_neighbor, i):
                return x_neighbor, True
        
        x_neighbor[i] = 0  # Reset
    
    return x0, False


# ============================================================================
# LOCAL SEARCH ALGORITHMS
# ============================================================================

def simple_descent_v1(A: np.ndarray, x0: np.ndarray, c: np.ndarray, 
                      verbose: bool = False) -> np.ndarray:
    """
    Simple descent with 2-1, 1-1, and 0-1 exchanges (sequential)
    
    Explores neighborhoods in order until no improvement found in any.
    
    Args:
        A: Constraint matrix
        x0: Initial solution
        c: Objective coefficients
        verbose: Print progress if True
        
    Returns:
        Locally optimal solution
    """
    x_improved = x0.copy()
    
    if verbose:
        print("Starting simple descent (2-1, 1-1, 0-1 exchanges)")
        print(f"Initial objective: {np.dot(x0, c)}")
    
    # Phase 1: 2-1 exchanges
    improved = True
    iterations = 0
    while improved:
        x_improved, improved = two_one_exchange(A, x_improved, c)
        if improved:
            iterations += 1
            if verbose:
                print(f"  2-1 exchange #{iterations}: obj = {np.dot(x_improved, c)}")
    
    if verbose:
        print(f"2-1 exchanges completed: {iterations} improvements")
    
    # Phase 2: 1-1 exchanges
    improved = True
    iterations = 0
    while improved:
        x_improved, improved = one_one_exchange(A, x_improved, c)
        if improved:
            iterations += 1
            if verbose:
                print(f"  1-1 exchange #{iterations}: obj = {np.dot(x_improved, c)}")
    
    if verbose:
        print(f"1-1 exchanges completed: {iterations} improvements")
    
    # Phase 3: 0-1 exchanges
    improved = True
    iterations = 0
    while improved:
        x_improved, improved = zero_one_exchange(A, x_improved, c)
        if improved:
            iterations += 1
            if verbose:
                print(f"  0-1 exchange #{iterations}: obj = {np.dot(x_improved, c)}")
    
    if verbose:
        print(f"0-1 exchanges completed: {iterations} improvements")
        print(f"Final objective: {np.dot(x_improved, c)}")
    
    return x_improved


def simple_descent_v2(A: np.ndarray, x0: np.ndarray, c: np.ndarray,
                      verbose: bool = False) -> np.ndarray:
    """
    Simple descent with 1-1 and 0-1 exchanges only
    
    Simpler and faster version, explores only 1-1 and 0-1 neighborhoods.
    
    Args:
        A: Constraint matrix
        x0: Initial solution
        c: Objective coefficients
        verbose: Print progress if True
        
    Returns:
        Locally optimal solution
    """
    x_improved = x0.copy()
    
    if verbose:
        print("Starting simple descent (1-1, 0-1 exchanges)")
        print(f"Initial objective: {np.dot(x0, c)}")
    
    # Phase 1: 1-1 exchanges
    improved = True
    iterations = 0
    while improved:
        x_improved, improved = one_one_exchange(A, x_improved, c)
        if improved:
            iterations += 1
            if verbose:
                print(f"  1-1 exchange #{iterations}: obj = {np.dot(x_improved, c)}")
    
    if verbose:
        print(f"1-1 exchanges completed: {iterations} improvements")
    
    # Phase 2: 0-1 exchanges
    improved = True
    iterations = 0
    while improved:
        x_improved, improved = zero_one_exchange(A, x_improved, c)
        if improved:
            iterations += 1
            if verbose:
                print(f"  0-1 exchange #{iterations}: obj = {np.dot(x_improved, c)}")
    
    if verbose:
        print(f"0-1 exchanges completed: {iterations} improvements")
        print(f"Final objective: {np.dot(x_improved, c)}")
    
    return x_improved