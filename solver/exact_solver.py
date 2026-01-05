# solver/exact_solver.py
import pulp
import time
from typing import Dict, Tuple, Optional

def solve_spp_exact(
    instance: Dict,
    solver_name: str = 'HiGHS',
    time_limit: int = 300,
    verbose: bool = False
) -> Dict:
    """
    Solve Set Packing Problem exactly using MIP solver
    
    SPP Formulation:
        Maximize: sum(c[j] * x[j]) for all sets j
        Subject to: sum(x[j] for j in sets containing item i) <= 1, for all items i
                   x[j] in {0, 1}
    
    Args:
        instance: Dictionary from load_instance with keys:
                 - n_items: number of items
                 - n_sets: number of sets
                 - c: coefficient vector (weights)
                 - A: constraint matrix
                 - sets: list of sets
        solver_name: 'HiGHS', 'GLPK', or 'CBC'
        time_limit: maximum solve time in seconds
        verbose: if True, print solver output
        
    Returns:
        Dictionary with:
            - solution: binary vector (selected sets)
            - objective: objective value
            - solve_time: time in seconds
            - status: solver status ('Optimal', 'Feasible', 'Infeasible', 'Timeout')
            - gap: optimality gap (if not optimal)
            - solver: solver used
    """
    n_items = instance['n_items']
    n_sets = instance['n_sets']
    weights = instance['weights']
    sets = instance['sets']
    
    # Create problem
    prob = pulp.LpProblem("SPP", pulp.LpMaximize)
    
    # Decision variables: x[j] = 1 if set j is selected, 0 otherwise
    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_sets)]
    
    # Objective function: maximize sum of weights of selected sets
    prob += pulp.lpSum([weights[j] * x[j] for j in range(n_sets)]), "Total_Weight"
    
    # Constraints: each item can appear in at most one selected set
    for item in range(1, n_items + 1):  # Items are 1-indexed in the data
        # Find all sets containing this item
        sets_with_item = [j for j in range(n_sets) if item in sets[j]]
        
        if sets_with_item:
            # Sum of selected sets containing this item must be <= 1
            prob += (
                pulp.lpSum([x[j] for j in sets_with_item]) <= 1,
                f"Item_{item}_constraint"
            )
    
    # Select and configure solver
    if solver_name == 'HiGHS':
        solver = pulp.HiGHS_CMD(
            msg=1 if verbose else 0,
            timeLimit=time_limit,
            options=[]
        )
    elif solver_name == 'GLPK':
        solver = pulp.GLPK_CMD(
            msg=1 if verbose else 0,
            options=['--tmlim', str(time_limit)]
        )
    elif solver_name == 'CBC':
        solver = pulp.PULP_CBC_CMD(
            msg=1 if verbose else 0,
            timeLimit=time_limit
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}. Use 'HiGHS', 'GLPK', or 'CBC'")
    
    # Solve
    start_time = time.time()
    prob.solve(solver)
    solve_time = time.time() - start_time
    
    # Extract solution
    solution = [int(var.varValue) if var.varValue is not None else 0 for var in x]
    
    # Calculate objective value
    objective = sum(weights[j] * solution[j] for j in range(n_sets))
    
    # Determine status
    status_map = {
        pulp.LpStatusOptimal: 'Optimal',
        pulp.LpStatusNotSolved: 'Not Solved',
        pulp.LpStatusInfeasible: 'Infeasible',
        pulp.LpStatusUnbounded: 'Unbounded',
        pulp.LpStatusUndefined: 'Undefined'
    }
    status = status_map.get(prob.status, 'Unknown')
    
    # Calculate optimality gap (if solver provides it)
    gap = None
    if status != 'Optimal' and objective > 0:
        # For non-optimal solutions, gap is unknown
        gap = None
    
    return {
        'solution': solution,
        'objective': objective,
        'solve_time': solve_time,
        'status': status,
        'gap': gap,
        'solver': solver_name,
        'n_items': n_items,
        'n_sets': n_sets
    }