# Set Packing Problem Solver

Optimization solver for the Set Packing Problem using exact methods and metaheuristics.

## Live Demo
🚀 [Coming soon - will deploy to Streamlit Cloud]

## Features
- **Exact Solver**: Optimization using PuLP (CBC/HiGHS/GLPK)
- **Greedy Construction**: Fast heuristic for initial solutions
- **Local Search**: k-p exchange neighborhoods (2-1, 1-1, 0-1)
- **Simple descent**: For complete neighborhood search

## Tech Stack
Python, NumPy, PuLP, Streamlit

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Problem Description
Set Packing: Select non-overlapping sets to maximize total weight.

Applications: project selection, task scheduling, resource allocation.