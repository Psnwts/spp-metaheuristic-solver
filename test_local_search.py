# test_local_search.py
import numpy as np
from utils.instance_loader import load_instance, list_instances
from solver.greedy import greedy_search, is_admissible
from solver.local_search import simple_descent_v1, simple_descent_v2
import time
import sys

def test_greedy_plus_local_search(filename, version=1):
    """
    Test greedy construction + local search on a single instance
    
    Args:
        filename: Name of the .dat file (e.g., 'instance1.dat')
        version: 1 for simple_descent_v1 (2-1, 1-1, 0-1 exchanges)
                 2 for simple_descent_v2 (1-1, 0-1 exchanges)
    """
    print("=" * 70)
    print(f"GREEDY + LOCAL SEARCH TEST - {filename}")
    print(f"Version: {'v1 (2-1, 1-1, 0-1)' if version == 1 else 'v2 (1-1, 0-1)'}")
    print("=" * 70)
    
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
    
    print("\n" + "=" * 70)
    print("PHASE 1: GREEDY CONSTRUCTION")
    print("=" * 70)
    
    # Run greedy construction
    print("\n🚀 Running greedy algorithm...")
    start_time = time.time()
    greedy_solution = greedy_search(instance['A'], instance['c'])
    greedy_time = time.time() - start_time
    
    # Greedy results
    greedy_feasible = is_admissible(instance['A'], greedy_solution)
    greedy_objective = instance['c'] @ greedy_solution
    greedy_selected = np.where(greedy_solution == 1)[0]
    
    print(f"✅ Greedy completed in {greedy_time:.4f} seconds")
    print(f"\n📊 Greedy Results:")
    print(f"  - Objective: {greedy_objective}")
    print(f"  - Feasible: {'✅ Yes' if greedy_feasible else '❌ No'}")
    print(f"  - Sets selected: {len(greedy_selected)}")
    print(f"  - Selected set indices: {list(greedy_selected)}")
    
    # Show greedy solution details
    print(f"\n📋 Greedy solution details:")
    for idx in greedy_selected[:5]:  # Show first 5
        print(f"  Set {idx}: weight={instance['weights'][idx]}, items={instance['sets'][idx]}")
    if len(greedy_selected) > 5:
        print(f"  ... and {len(greedy_selected) - 5} more sets")
    
    print("\n" + "=" * 70)
    print(f"PHASE 2: LOCAL SEARCH (VERSION {version})")
    print("=" * 70)
    
    # Run local search
    if version == 1:
        print("\n🔍 Running local search with 2-1, 1-1, 0-1 exchanges...")
        start_time = time.time()
        improved_solution = simple_descent_v1(
            instance['A'], 
            greedy_solution, 
            instance['c'],
            verbose=True  # Show progress
        )
        local_search_time = time.time() - start_time
    else:
        print("\n🔍 Running local search with 1-1, 0-1 exchanges...")
        start_time = time.time()
        improved_solution = simple_descent_v2(
            instance['A'], 
            greedy_solution, 
            instance['c'],
            verbose=True  # Show progress
        )
        local_search_time = time.time() - start_time
    
    # Local search results
    improved_feasible = is_admissible(instance['A'], improved_solution)
    improved_objective = instance['c'] @ improved_solution
    improved_selected = np.where(improved_solution == 1)[0]
    
    print(f"\n✅ Local search completed in {local_search_time:.4f} seconds")
    
    print(f"\n📊 Local Search Results:")
    print(f"  - Objective: {improved_objective}")
    print(f"  - Feasible: {'✅ Yes' if improved_feasible else '❌ No'}")
    print(f"  - Sets selected: {len(improved_selected)}")
    print(f"  - Selected set indices: {list(improved_selected)}")
    
    # Show improved solution details
    print(f"\n📋 Improved solution details:")
    for idx in improved_selected[:5]:  # Show first 5
        print(f"  Set {idx}: weight={instance['weights'][idx]}, items={instance['sets'][idx]}")
    if len(improved_selected) > 5:
        print(f"  ... and {len(improved_selected) - 5} more sets")
    
    print("\n" + "=" * 70)
    print("COMPARISON: GREEDY vs GREEDY + LOCAL SEARCH")
    print("=" * 70)
    
    # Calculate improvement
    improvement = improved_objective - greedy_objective
    improvement_pct = (improvement / greedy_objective * 100) if greedy_objective > 0 else 0
    total_time = greedy_time + local_search_time
    
    print(f"\n📈 Improvement Analysis:")
    print(f"  - Greedy objective:       {greedy_objective}")
    print(f"  - After local search:     {improved_objective}")
    print(f"  - Improvement:            +{improvement} ({improvement_pct:+.2f}%)")
    
    print(f"\n⏱️  Time Analysis:")
    print(f"  - Greedy time:            {greedy_time:.4f}s")
    print(f"  - Local search time:      {local_search_time:.4f}s")
    print(f"  - Total time:             {total_time:.4f}s")
    
    print(f"\n📦 Solution Changes:")
    print(f"  - Greedy selected:        {len(greedy_selected)} sets")
    print(f"  - After local search:     {len(improved_selected)} sets")
    
    # Show which sets were added/removed
    added_sets = set(improved_selected) - set(greedy_selected)
    removed_sets = set(greedy_selected) - set(improved_selected)
    
    if added_sets:
        print(f"  - Sets added:             {sorted(added_sets)}")
    if removed_sets:
        print(f"  - Sets removed:           {sorted(removed_sets)}")
    if not added_sets and not removed_sets:
        print(f"  - No changes (local optimum reached by greedy)")
    
    # Items coverage comparison
    greedy_items = set()
    for idx in greedy_selected:
        greedy_items.update(instance['sets'][idx])
    
    improved_items = set()
    for idx in improved_selected:
        improved_items.update(instance['sets'][idx])
    
    print(f"\n📋 Coverage Analysis:")
    print(f"  - Greedy coverage:        {len(greedy_items)}/{instance['n_items']} items ({len(greedy_items)/instance['n_items']*100:.1f}%)")
    print(f"  - After local search:     {len(improved_items)}/{instance['n_items']} items ({len(improved_items)/instance['n_items']*100:.1f}%)")
    
    # Verification
    print(f"\n🔍 Solution Verification:")
    if greedy_feasible and improved_feasible:
        print(f"  ✅ Both solutions are feasible")
    elif not greedy_feasible:
        print(f"  ❌ WARNING: Greedy solution is infeasible!")
    elif not improved_feasible:
        print(f"  ❌ WARNING: Local search solution is infeasible!")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if improvement > 0:
        print(f"\n✅ Local search improved the solution!")
        print(f"   Objective increased by {improvement} ({improvement_pct:.2f}%)")
        print(f"   Total solve time: {total_time:.4f}s")
    else:
        print(f"\n✓  Greedy solution was already locally optimal")
        print(f"   No improvement found by local search")
        print(f"   Total solve time: {total_time:.4f}s")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


def compare_both_versions(filename):
    """
    Compare both local search versions on the same instance
    
    Args:
        filename: Name of the .dat file
    """
    print("=" * 70)
    print(f"COMPARING LOCAL SEARCH VERSIONS - {filename}")
    print("=" * 70)
    
    # Load instance
    try:
        instance = load_instance(filename)
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found!")
        return
    
    print(f"\n📋 Instance: {instance['n_items']} items, {instance['n_sets']} sets")
    
    # Run greedy once
    print("\n🚀 Running greedy construction...")
    greedy_solution = greedy_search(instance['A'], instance['c'])
    greedy_objective = instance['c'] @ greedy_solution
    print(f"   Greedy objective: {greedy_objective}")
    
    print("\n" + "-" * 70)
    
    # Test version 1
    print("\n🔍 Testing Version 1 (2-1, 1-1, 0-1 exchanges)...")
    start_time = time.time()
    solution_v1 = simple_descent_v1(instance['A'], greedy_solution, instance['c'], verbose=False)
    time_v1 = time.time() - start_time
    obj_v1 = instance['c'] @ solution_v1
    improvement_v1 = obj_v1 - greedy_objective
    
    print(f"   Objective: {obj_v1} (+{improvement_v1})")
    print(f"   Time: {time_v1:.4f}s")
    
    print("\n" + "-" * 70)
    
    # Test version 2
    print("\n🔍 Testing Version 2 (1-1, 0-1 exchanges)...")
    start_time = time.time()
    solution_v2 = simple_descent_v2(instance['A'], greedy_solution, instance['c'], verbose=False)
    time_v2 = time.time() - start_time
    obj_v2 = instance['c'] @ solution_v2
    improvement_v2 = obj_v2 - greedy_objective
    
    print(f"   Objective: {obj_v2} (+{improvement_v2})")
    print(f"   Time: {time_v2:.4f}s")
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Objective':<15} {'Improvement':<15} {'Time (s)':<15}")
    print("-" * 70)
    print(f"{'Greedy':<25} {greedy_objective:<15} {'-':<15} {'-':<15}")
    print(f"{'Version 1 (2-1,1-1,0-1)':<25} {obj_v1:<15} {f'+{improvement_v1}':<15} {time_v1:<15.4f}")
    print(f"{'Version 2 (1-1,0-1)':<25} {obj_v2:<15} {f'+{improvement_v2}':<15} {time_v2:<15.4f}")
    
    print("\n📊 Analysis:")
    if obj_v1 > obj_v2:
        print(f"   ✅ Version 1 found better solution (+{obj_v1 - obj_v2} over v2)")
        print(f"   ⏱️  But took {time_v1/time_v2:.2f}x longer")
    elif obj_v2 > obj_v1:
        print(f"   ✅ Version 2 found better solution (+{obj_v2 - obj_v1} over v1)")
        print(f"   ⚡ And was {time_v1/time_v2:.2f}x faster")
    else:
        print(f"   ✓  Both versions found same solution quality")
        if time_v2 < time_v1:
            print(f"   ⚡ Version 2 was {time_v1/time_v2:.2f}x faster")
        else:
            print(f"   Version 1 was {time_v2/time_v1:.2f}x faster")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_local_search.py <filename> [version]")
        print("  python test_local_search.py <filename> compare")
        print("\nExamples:")
        print("  python test_local_search.py didactic.dat 1")
        print("  python test_local_search.py didactic.dat 2")
        print("  python test_local_search.py didactic.dat compare")
        print("\nAvailable instances:")
        for inst in list_instances():
            print(f"  - {inst}")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if len(sys.argv) >= 3:
        if sys.argv[2] == "compare":
            compare_both_versions(filename)
        else:
            version = int(sys.argv[2])
            test_greedy_plus_local_search(filename, version)
    else:
        # Default to version 1
        test_greedy_plus_local_search(filename, version=1)