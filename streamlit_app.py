"""Streamlit GUI for the multi-stage crime analysis pipeline.

Usage:
    streamlit run streamlit_app.py

This UI delegates all logic to run_pipeline.run_full_pipeline.
"""

import os
import streamlit as st

from run_pipeline import run_full_pipeline

st.set_page_config(page_title="Crime Analysis Pipeline", layout="wide")

st.title("Crime Analysis Pipeline (LLM-1 to LLM-4)")

st.markdown(
    """
This app runs the full multi-stage pipeline:
LLM-1 (motivation) → LLM-2 (historical) → LLM-3 (pattern) → Fusion → LLM-4 (report).
Provide a crime description and optional context. Missing fields are handled safely.
"""
)

with st.sidebar:
    st.header("Configuration")
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("Set GEMINI_API_KEY env var for live Gemini calls. Without it, fallbacks are used.")

st.subheader("Input")
crime_text = st.text_area("Crime description (required)", height=150)

col1, col2, col3 = st.columns(3)
with col1:
    state = st.text_input("State (optional)")
    premis_desc = st.text_input("Premises / location type", value="Unknown")
with col2:
    year = st.number_input("Year (optional)", min_value=1900, max_value=2100, value=2020, step=1)
    weapon_desc = st.text_input("Weapon description", value="Unknown")
with col3:
    crime_type = st.text_input("Crime type", value="Unknown")
    district = st.text_input("District", value="Unknown")

domestic = st.selectbox("Domestic case?", ["Unknown", "true", "false"])
arrest = st.selectbox("Arrest made?", ["Unknown", "true", "false"])
vict_age = st.text_input("Victim age", value="Unknown")
vict_sex = st.text_input("Victim sex", value="Unknown")
area_name = st.text_input("Area / locality", value="Unknown")
status_desc = st.text_input("Status description", value="Unknown")

if st.button("Analyze Crime", type="primary"):
    if not crime_text.strip():
        st.error("Crime description is required.")
    else:
        input_record = {
            "crime_text": crime_text,
            "state": state or None,
            "year": int(year) if year else None,
            "crime_type": crime_type,
            "premis_desc": premis_desc,
            "weapon_desc": weapon_desc,
            "district": district,
            "domestic": domestic,
            "arrest": arrest,
            "vict_age": vict_age,
            "vict_sex": vict_sex,
            "area_name": area_name,
            "status_desc": status_desc,
        }

        with st.spinner("Running full pipeline..."):
            result = run_full_pipeline(input_record)

        st.success("Pipeline complete.")

        st.markdown("### LLM-1 Motivation")
        st.json(result.get("llm1", {}))

        st.markdown("### LLM-2 Historical")
        st.json(result.get("llm2", {}))

        st.markdown("### LLM-3 Pattern")
        st.json(result.get("llm3", {}))

        st.markdown("### Fusion Output")
        st.json(result.get("fusion", {}))

        st.markdown("### LLM-4 Report")
        st.write(result.get("report", ""))
