import streamlit as st

# -----------------------------
# YOUR EXISTING PIPELINE IMPORT
# -----------------------------
# If run_pipeline is in another file:
# from run_pipeline import run_pipeline

# TEMP EXAMPLE (replace with yours)
def run_pipeline(text):
    # Replace this with your real function
    return {
        "LLM1": "Possible financial motivation",
        "LLM2": "Pattern resembles opportunistic crime",
        "LLM3": "Context indicates planned action",
        "Fusion": "Likely financial + situational motive",
        "Final Report": "The suspect appears driven by financial need with premeditation."
    }


# -----------------------------
# STREAMLIT GUI
# -----------------------------

st.set_page_config(page_title="Criminal Mind Analysis", layout="wide")

st.title("🧠 Criminal Mind Analysis — Research GUI")

st.markdown("Enter a case description and generate an AI analysis report.")

case_text = st.text_area(
    "Case Description",
    height=200,
    placeholder="Example: A person entered a store at night and threatened the cashier..."
)

if st.button("Generate Report"):

    if case_text.strip() == "":
        st.warning("Please enter a case description.")
    else:
        with st.spinner("Running analysis..."):

            result = run_pipeline(case_text)

        st.success("Analysis complete.")

        st.subheader("📊 LLM Outputs")

        st.markdown("### LLM1")
        st.write(result["LLM1"])

        st.markdown("### LLM2")
        st.write(result["LLM2"])

        st.markdown("### LLM3")
        st.write(result["LLM3"])

        st.markdown("### 🔗 Fusion Layer")
        st.write(result["Fusion"])

        st.markdown("### 📄 Final Report")
        st.info(result["Final Report"])
