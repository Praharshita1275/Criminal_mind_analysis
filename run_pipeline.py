# ==========================================
# RUN PIPELINE — MASTER FUNCTION
# ==========================================
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-pro")


def run_pipeline(case_text):
    """
    Runs full Criminal Mind Analysis pipeline.
    Input: case description text
    Output: dictionary containing full report
    """

    # -----------------------------
    # STEP 1 — LLM ANALYSIS
    # -----------------------------
    llm1_out = analyze_with_llm1(case_text)
    llm2_out = analyze_with_llm2(case_text)
    llm3_out = analyze_with_llm3(case_text)

    # -----------------------------
    # STEP 2 — FUSION LAYER
    # -----------------------------
    fused_output = fuse_outputs(
        llm1_out,
        llm2_out,
        llm3_out
    )

    # -----------------------------
    # STEP 3 — FINAL REASONING
    # -----------------------------
    final_report = llm4_final_reasoning(fused_output)

    # -----------------------------
    # STEP 4 — RETURN EVERYTHING
    # -----------------------------
    return {
        "LLM1": llm1_out,
        "LLM2": llm2_out,
        "LLM3": llm3_out,
        "Fusion": fused_output,
        "Final Report": final_report
    }
