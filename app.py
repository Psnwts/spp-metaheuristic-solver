# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from utils.instance_loader import load_instance, list_instances
from solver.exact_solver import solve_spp_exact
from solver.greedy import greedy_search, is_admissible
from solver.local_search import simple_descent_v1, simple_descent_v2

st.set_page_config(page_title="SPP Solver", page_icon="🎯", layout="wide")

st.title("🎯 Set Packing Problem Solver")
st.markdown("Optimization solver using exact methods and metaheuristics")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Instance selection
st.sidebar.subheader("📋 Instance")
instances = list_instances()

if not instances:
    st.error("❌ No instances found in data/ folder!")
    st.stop()

selected_file = st.sidebar.selectbox("Select instance:", instances)
instance = load_instance(selected_file)

# Show instance info
with st.sidebar.expander("ℹ️ Instance Details"):
    st.metric("Items", instance['n_items'])
    st.metric("Sets", instance['n_sets'])
    st.metric("Density", f"{np.mean(instance['A']):.1%}")
    st.metric("Total Weight", sum(instance['weights']))

st.sidebar.divider()

# Solver selection
st.sidebar.subheader("🔧 Solver")
solver_choice = st.sidebar.radio(
    "Choose solver:",
    ["Exact (PuLP)", "Greedy + Local Search"],
    help="Exact guarantees optimality, Greedy+Local Search is faster"
)

# Solver-specific parameters
if solver_choice == "Exact (PuLP)":
    exact_solver = st.sidebar.selectbox("Solver:", ["CBC", "HiGHS", "GLPK"])
    time_limit = st.sidebar.slider("Time limit (s):", 10, 300, 60)
    
else:  # Greedy + Local Search
    descent_version = st.sidebar.radio(
        "Local search variant:",
        ["Version 1: (2-1, 1-1, 0-1)", "Version 2: (1-1, 0-1)"],
        help="Version 1 uses more neighborhoods, Version 2 is faster"
    )
    version = 1 if "Version 1" in descent_version else 2
    
    verbose = st.sidebar.checkbox("Show iterations", value=False)
    time_limit = st.sidebar.number_input("Time limit (s):", 1, 300, 60)

# Main content
st.header(f"📊 Problem Instance: {selected_file}")

# Instance metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 Items", instance['n_items'])
with col2:
    st.metric("📚 Sets", instance['n_sets'])
with col3:
    st.metric("📊 Density", f"{np.mean(instance['A']):.1%}")
with col4:
    st.metric("⚖️ Total Weight", sum(instance['weights']))

st.divider()

# Solve button
if st.button("▶️ Run Solver", type="primary", use_container_width=True):
    
    if solver_choice == "Exact (PuLP)":
        # ============================================================
        # EXACT SOLVER
        # ============================================================
        
        with st.spinner(f"Solving with {exact_solver}..."):
            result = solve_spp_exact(
                instance,
                solver_name=exact_solver,
                time_limit=time_limit,
                verbose=False
            )
        
        # Status indicator
        if result['status'] == 'Optimal':
            st.success(f"✅ Optimal solution found!")
        else:
            st.warning(f"⚠️ Status: {result['status']}")
        
        # Results
        st.header("📊 Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Objective", result['objective'])
        with col2:
            st.metric("⏱️ Time (s)", f"{result['solve_time']:.3f}")
        with col3:
            selected_count = sum(result['solution'])
            st.metric("📚 Sets Selected", selected_count)
        
        # Verification
        is_feasible, error = verify_solution(instance, result['solution'])
        if is_feasible:
            st.success("✅ Solution verified as feasible")
        else:
            st.error(f"❌ Verification failed: {error}")
        
        # Selected sets
        st.subheader("📦 Selected Sets")
        selected_indices = [j for j, val in enumerate(result['solution']) if val == 1]
        
        if selected_indices:
            solution_df = pd.DataFrame({
                'Set': selected_indices,
                'Weight': [instance['weights'][j] for j in selected_indices],
                'Items': [str(instance['sets'][j]) for j in selected_indices]
            })
            st.dataframe(solution_df, use_container_width=True)
        else:
            st.info("No sets selected")
        
        # Visualization
        st.subheader("📈 Solution Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['green' if result['solution'][j] == 1 else 'lightgray' 
                 for j in range(instance['n_sets'])]
        ax.bar(range(instance['n_sets']), instance['weights'], color=colors)
        ax.set_xlabel('Set Index')
        ax.set_ylabel('Weight')
        ax.set_title('Set Weights (Green = Selected)')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
    
    else:
        # ============================================================
        # GREEDY + LOCAL SEARCH
        # ============================================================
        
        # Phase 1: Greedy Construction
        st.subheader("🔨 Phase 1: Greedy Construction")
        
        with st.spinner("Running greedy algorithm..."):
            start_time = time.time()
            greedy_solution = greedy_search(instance['A'], instance['c'])
            greedy_time = time.time() - start_time
        
        greedy_obj = instance['c'] @ greedy_solution
        greedy_feasible = is_admissible(instance['A'], greedy_solution)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Objective", greedy_obj)
        with col2:
            st.metric("Time (s)", f"{greedy_time:.4f}")
        with col3:
            st.metric("Feasible", "✅" if greedy_feasible else "❌")
        
        # Phase 2: Local Search
        st.subheader("🔍 Phase 2: Local Search")
        
        # Capture verbose output
        if verbose:
            progress_placeholder = st.empty()
            
            # Monkey-patch print to capture output (simple approach)
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
        
        with st.spinner(f"Running local search (Version {version})..."):
            search_start = time.time()
            
            # Check time limit
            if time.time() - start_time > time_limit:
                st.warning(f"⏱️ Time limit reached during greedy phase")
                improved_solution = greedy_solution
                local_search_time = 0
            else:
                if version == 1:
                    improved_solution = simple_descent_v1(
                        instance['A'], 
                        greedy_solution, 
                        instance['c'],
                        verbose=verbose
                    )
                else:
                    improved_solution = simple_descent_v2(
                        instance['A'], 
                        greedy_solution, 
                        instance['c'],
                        verbose=verbose
                    )
                
                local_search_time = time.time() - search_start
        
        # Restore stdout and show captured output
        if verbose:
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            if output:
                with st.expander("📝 Iteration Log"):
                    st.text(output)
        
        improved_obj = instance['c'] @ improved_solution
        improved_feasible = is_admissible(instance['A'], improved_solution)
        improvement = improved_obj - greedy_obj
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Objective", improved_obj, delta=f"+{improvement}")
        with col2:
            st.metric("Time (s)", f"{local_search_time:.4f}")
        with col3:
            st.metric("Feasible", "✅" if improved_feasible else "❌")
        
        # Summary
        st.divider()
        st.header("📊 Summary")
        
        total_time = greedy_time + local_search_time
        improvement_pct = (improvement / greedy_obj * 100) if greedy_obj > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Results Comparison")
            comparison_df = pd.DataFrame({
                'Method': ['Greedy', 'After Local Search', 'Improvement'],
                'Objective': [greedy_obj, improved_obj, improvement],
                'Time (s)': [f"{greedy_time:.4f}", f"{local_search_time:.4f}", f"{total_time:.4f}"]
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            if improvement > 0:
                st.success(f"✅ Local search improved solution by {improvement} ({improvement_pct:.1f}%)")
            else:
                st.info("✓ Greedy solution was already locally optimal")
        
        with col2:
            st.subheader("Solution Quality")
            
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(6, 4))
            methods = ['Greedy', 'Greedy +\nLocal Search']
            objectives = [greedy_obj, improved_obj]
            colors = ['skyblue', 'green']
            
            bars = ax.bar(methods, objectives, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Objective Value')
            ax.set_title('Solution Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        # Selected sets
        st.divider()
        st.subheader("📦 Final Solution")
        
        selected_indices = np.where(improved_solution == 1)[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if len(selected_indices) > 0:
                solution_df = pd.DataFrame({
                    'Set': selected_indices,
                    'Weight': [instance['weights'][j] for j in selected_indices],
                    'Items': [str(instance['sets'][j]) for j in selected_indices]
                })
                st.dataframe(solution_df, use_container_width=True)
            else:
                st.info("No sets selected")
        
        with col2:
            # Coverage info
            items_covered = set()
            for idx in selected_indices:
                items_covered.update(instance['sets'][idx])
            
            coverage_pct = (len(items_covered) / instance['n_items']) * 100
            
            st.metric("Sets Selected", len(selected_indices))
            st.metric("Items Covered", f"{len(items_covered)}/{instance['n_items']}")
            st.metric("Coverage", f"{coverage_pct:.1f}%")

# Footer
st.divider()
st.caption("🎯 SPP Solver | Greedy Construction + k-p Exchange Local Search")