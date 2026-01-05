# test_greedy.py
import numpy as np
from utils.instance_loader import load_instance, list_instances
from solver.greedy import greedy_search, is_admissible
import time
import sys

def test_greedy_solver():
    """Test greedy solver on an instance with detailed output"""
    print("=" * 60)
    print("SPP GREEDY SOLVER - TEST RUN")
    print("=" * 60)
    
    # List available instances
    print("\n📁 Available instances:")
    instances = list_instances()
    
    if not instances:
        print("❌ No instances found in data/ folder!")
        return
    
    for i, inst in enumerate(instances):
        print(f"  {i+1}. {inst}")
    
    # Select instance (you can modify this to select different ones)
    selected_file = instances[0]  # Using first instance
    print(f"\n✅ Selected: {selected_file}")
    
    # Load instance
    print("\n📋 Loading instance...")
    instance = load_instance(selected_file)
    
    print(f"  - Items (resources): {instance['n_items']}")
    print(f"  - Sets (candidates): {instance['n_sets']}")
    print(f"  - Total weight: {sum(instance['weights'])}")
    print(f"  - Matrix density: {np.mean(instance['A']):.2%}")
    
    # Show weights
    print(f"\n⚖️  Set weights: {instance['weights']}")
    
    # Show first few sets
    print("\n📚 First 5 sets:")
    for i in range(min(5, instance['n_sets'])):
        print(f"  Set {i}: weight={instance['weights'][i]}, items={instance['sets'][i]}")
    
    print("\n" + "-" * 60)
    print("🚀 RUNNING GREEDY ALGORITHM")
    print("-" * 60)
    
    # Run greedy search
    start_time = time.time()
    solution = greedy_search(instance['A'], instance['c'])
    solve_time = time.time() - start_time
    
    print(f"\n✅ Greedy search completed in {solve_time:.4f} seconds")
    
    # Check feasibility
    print("\n🔍 Checking feasibility...")
    feasible = is_admissible(instance['A'], solution)
    
    if feasible:
        print("  ✅ Solution is FEASIBLE")
    else:
        print("  ❌ Solution is INFEASIBLE")
    
    # Calculate objective
    objective = instance['c'] @ solution
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Objective value: {objective}")
    print(f"⏱️  Solve time: {solve_time:.4f} seconds")
    print(f"✓  Feasibility: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    
    # Solution details
    selected_sets = np.where(solution == 1)[0]
    print(f"\n📦 Number of selected sets: {len(selected_sets)}")
    print(f"📦 Selected set indices: {list(selected_sets)}")
    
    # Show selected sets details
    print("\n📋 Selected sets details:")
    total_weight = 0
    for idx in selected_sets:
        weight = instance['weights'][idx]
        items = instance['sets'][idx]
        total_weight += weight
        print(f"  Set {idx}: weight={weight}, items={items}")
    
    print(f"\n💰 Total weight (check): {total_weight}")
    
    # Items coverage
    items_used = set()
    for idx in selected_sets:
        items_used.update(instance['sets'][idx])
    
    coverage_pct = (len(items_used) / instance['n_items']) * 100
    
    print(f"\n📋 Items coverage:")
    print(f"  - Items covered: {len(items_used)}/{instance['n_items']}")
    print(f"  - Coverage: {coverage_pct:.1f}%")
    print(f"  - Covered items: {sorted(items_used)}")
    
    # Solution vector
    print(f"\n🔢 Solution vector (x):")
    print(f"  {solution}")
    
    # Verification
    print("\n🔍 Detailed verification:")
    item_usage = instance['A'] @ solution
    print(f"  Item usage (A @ x): {item_usage}")
    print(f"  Max item usage: {np.max(item_usage)}")
    
    if np.max(item_usage) > 1:
        print("  ⚠️  WARNING: Some items used more than once!")
        violated_items = np.where(item_usage > 1)[0]
        print(f"  Violated items: {violated_items + 1}")  # +1 for 1-indexing
    else:
        print("  ✅ All items used at most once")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

def test_single_instance(filename):
    """
    Test greedy solver on a single instance
    
    Args:
        filename: Name of the .dat file (e.g., 'instance1.dat')
    """
    print("=" * 60)
    print(f"SPP GREEDY SOLVER - {filename}")
    print("=" * 60)
    
    # Load instance
    try:
        print("\n📋 Loading instance...")
        instance = load_instance(filename)
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found in data/ folder!")
        print("\nAvailable instances:")
        for inst in list_instances():
            print(f"  - {inst}")
        return
    
    print(f"  - Items (resources): {instance['n_items']}")
    print(f"  - Sets (candidates): {instance['n_sets']}")
    print(f"  - Total weight: {sum(instance['weights'])}")
    print(f"  - Matrix density: {np.mean(instance['A']):.2%}")
    
    # Show weights
    print(f"\n⚖️  Set weights: {instance['weights']}")
    
    print("\n" + "-" * 60)
    print("🚀 RUNNING GREEDY ALGORITHM")
    print("-" * 60)
    
    # Run greedy search
    start_time = time.time()
    solution = greedy_search(instance['A'], instance['c'])
    solve_time = time.time() - start_time
    
    print(f"\n✅ Greedy search completed in {solve_time:.4f} seconds")
    
    # Check feasibility
    print("\n🔍 Checking feasibility...")
    feasible = is_admissible(instance['A'], solution)
    
    if feasible:
        print("  ✅ Solution is FEASIBLE")
    else:
        print("  ❌ Solution is INFEASIBLE")
    
    # Calculate objective
    objective = instance['c'] @ solution
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Objective value: {objective}")
    print(f"⏱️  Solve time: {solve_time:.4f} seconds")
    print(f"🔢 Objective solution: {solution}")
    print(f"✓  Feasibility: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    


if __name__ == "__main__":
    # Check if filename provided as command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        test_single_instance(filename)
    else:
        test_greedy_solver()