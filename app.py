import streamlit as st
from utils.instance_loader import load_instance,list_instances,get_instance_info



st.title("SPP Solver")

# List available instances
instances = list_instances()
selected_file = st.selectbox("Select instance", instances)

# Load instance
instance = load_instance(selected_file)
st.info(get_instance_info(instance))
st.info(instance["c"])

# Solve button
if st.button("Solve"):
    solver = 1
    solution = 2
    st.success(f"Objective: {000}")

