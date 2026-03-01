
from typing import Dict
import streamlit as st

def hpo_checkbox_panel(hpo_mapping: dict) -> Dict[str, int]:
    st.subheader("Symptoms (HPO flags)")
    flags = {}
    for hpo_col, meta in hpo_mapping.items():
        label = meta.get("label", hpo_col)
        checked = st.checkbox(f"{label} ({hpo_col})", value=False)
        flags[hpo_col] = 1 if checked else 0
    return flags