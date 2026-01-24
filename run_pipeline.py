"""End-to-end crime analysis pipeline using multiple LLM agents and a fusion layer.

Layers
- LLM-1: Motivation agent (Gemini). Input: single crime record (preprocessed).
- LLM-2: Historical motivation agent (Gemini). Input: {state, year, crime_type}.
- LLM-3: Crime pattern agent (Gemini). Input: context record with location/pattern info.
- Fusion: Deterministic logic. No LLM calls.
- LLM-4: Final report agent (Gemini). Input: fusion output only.

Notes
- No data preprocessing is done here. Uses already-preprocessed files:
  processed_crime_data.csv
  crime_data_llm_ready.json
  llm2_motivation_dataset.json
  llm3_context_dataset.json
- Gemini API key is read from the GEMINI_API_KEY environment variable.
- All Gemini calls are wrapped with defensive error handling and JSON parsing safeguards.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
except Exception:  # pragma: no cover - optional dependency
    genai = None
    HarmBlockThreshold = None
    HarmCategory = None

# ---------------------------------------------------------------------------
# Constants and configs
# ---------------------------------------------------------------------------
CONF_WEIGHT = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
PATTERN_MOTIVATION_MAP = {
    "domestic violence pattern": "emotional",
    "public/street crime pattern": "financial",
    "weapon escalation pattern": "power",
    "opportunistic crime pattern": "financial",
    "organized/repeat crime pattern": "financial",
    "unclear/general pattern": "unknown",
}
DEFAULT_CONFIDENCE = "Low"
DEFAULT_MOTIVATION = "unknown"

DATA_FILES = {
    "llm2": "llm2_motivation_dataset.json",
    "llm3": "llm3_context_dataset.json",
    "csv_processed": "processed_crime_data.csv",
    "json_processed": "crime_data_llm_ready.json",
}

LLM1_SYSTEM_PROMPT = """
You are an intelligent crime motivation analysis agent.

Your task:
1. Predict the most likely motivation behind a crime.
2. Explain the reasoning based only on available information.
3. Provide a confidence level (Low / Medium / High).

Rules:
- Input data comes from preprocessed crime datasets.
- Some fields may be missing or marked as Unknown.
- Do NOT assume missing details.
- Possible motivations: emotional, financial, power, sexual, unknown.
- Respond ONLY in JSON format with keys: predicted_motivation, reasoning, confidence.
"""

LLM2_SYSTEM_PROMPT = """
You are a crime analytics agent specializing in historical and regional crime motivation analysis.

Your task:
1. Analyze historical motivation patterns using provided data.
2. Explain which motivations are dominant and why.
3. Relate historical trends to the given case context.
4. Handle missing state or year gracefully.
5. Never invent numerical values.

Possible motivations include: emotional, financial, power, sexual, personal vendetta, political, other, unknown.

Respond ONLY in JSON format with keys: dominant_historical_motivation, explanation, confidence_level.
"""

LLM3_SYSTEM_PROMPT = """
You are a crime pattern analysis agent.

Your task:
1. Identify the crime context pattern based on structured information.
2. Explain what kind of behavioral or situational pattern this crime belongs to.
3. Handle missing or unknown fields gracefully.
4. Do not infer motivation (that is handled by another model).

Possible patterns include:
- domestic violence pattern
- public/street crime pattern
- weapon escalation pattern
- opportunistic crime pattern
- organized/repeat crime pattern
- unclear/general pattern

Respond ONLY in JSON format with keys: identified_pattern, explanation, confidence_level.
"""

LLM4_SYSTEM_PROMPT = """
You are an expert crime analysis report agent.

Your task:
1. Generate a clear analytical report.
2. Use ONLY the provided fused model output.
3. Do NOT introduce new facts or assumptions.
4. Explain what happened, why it likely happened, and confidence level.
5. Keep reasoning logical and evidence-based.

Respond in structured text (not JSON).
"""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_json(path: str, fallback: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return fallback
    except json.JSONDecodeError:
        return fallback


def _safe_str(value: Any, default: str = "Unknown") -> str:
    if value is None:
        return default
    try:
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _confidence_to_weight(level: str) -> float:
    return CONF_WEIGHT.get(level, CONF_WEIGHT[DEFAULT_CONFIDENCE])


@dataclass
class GeminiClient:
    model_name: str = "gemini-1.5-flash"

    def __post_init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.enabled = bool(genai and self.api_key)
        if not self.enabled:
            return
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
        except Exception:
            self.enabled = False

    def generate_json(self, system_prompt: str, user_prompt: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Call Gemini and parse JSON. Returns (dict, raw_text)."""
        if not self.enabled:
            return {}, None
        raw_text = None
        try:
            response = self.model.generate_content(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            raw_text = (response.text or "").strip()
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed, raw_text
            return {}, raw_text
        except Exception:
            # Any failure returns empty dict with raw text for debugging
            return {}, raw_text

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.enabled:
            return "Gemini unavailable; no report generated."
        try:
            response = self.model.generate_content(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return (response.text or "").strip()
        except Exception:
            return "Error generating report."


# ---------------------------------------------------------------------------
# Load datasets once
# ---------------------------------------------------------------------------
LLM2_DATA = _load_json(DATA_FILES["llm2"], [])
LLM3_DATA = _load_json(DATA_FILES["llm3"], [])
PROCESSED_JSON = _load_json(DATA_FILES["json_processed"], [])
try:
    PROCESSED_DF = pd.read_csv(DATA_FILES["csv_processed"])
except Exception:
    PROCESSED_DF = pd.DataFrame()

# ---------------------------------------------------------------------------
# LLM agent functions
# ---------------------------------------------------------------------------

def llm1_analyze(record: Dict[str, Any], client: GeminiClient) -> Dict[str, Any]:
    """LLM-1 motivation agent using Gemini with strict JSON contract."""
    crime_text = _safe_str(record.get("crime_text"), "Not available")
    user_prompt = f"""
Crime Case (Preprocessed Record):

Crime Description:
{crime_text}

Crime Type: {_safe_str(record.get("crime_type"))}
Weapon Used: {_safe_str(record.get("weapon_desc"))}
Location Type: {_safe_str(record.get("premis_desc"))}
Victim Age: {_safe_str(record.get("vict_age"))}
Victim Sex: {_safe_str(record.get("vict_sex"))}
Area: {_safe_str(record.get("area_name"))}
Domestic Case: {_safe_str(record.get("domestic"))}
Arrest Status: {_safe_str(record.get("status_desc"))}

Analyze this case and predict the crime motivation.
"""

    parsed, raw = client.generate_json(LLM1_SYSTEM_PROMPT, user_prompt)
    if not parsed:
        return {
            "llm_stage": "LLM-1",
            "predicted_motivation": DEFAULT_MOTIVATION,
            "confidence": DEFAULT_CONFIDENCE,
            "reasoning": "Fallback: Gemini unavailable or invalid JSON.",
            "raw_response": raw,
        }

    return {
        "llm_stage": "LLM-1",
        "predicted_motivation": parsed.get("predicted_motivation", DEFAULT_MOTIVATION),
        "confidence": parsed.get("confidence", DEFAULT_CONFIDENCE),
        "reasoning": parsed.get("reasoning", ""),
        "raw_response": raw,
    }


def llm2_analyze(context: Dict[str, Any], client: GeminiClient) -> Dict[str, Any]:
    """LLM-2 historical motivation agent."""
    state = context.get("state")
    year = context.get("year")
    crime_type = context.get("crime_type")

    matched_record = None
    if state and year:
        for rec in LLM2_DATA:
            try:
                if rec.get("state") == state and int(rec.get("year", -1)) == int(year):
                    matched_record = rec
                    break
            except Exception:
                continue

    if matched_record:
        historical_text = json.dumps(matched_record.get("motivation_distribution", {}), indent=2)
        note = "Exact historical data found."
    else:
        historical_text = "Exact state/year data not available. Use generalized trends."
        note = "Using generalized historical trends."

    user_prompt = f"""
Crime Context (Preprocessed):
State: {_safe_str(state)}
Year: {_safe_str(year)}
Crime Type: {_safe_str(crime_type)}

Historical Motivation Data:
{historical_text}

Note: {note}

Analyze dominant historical motivations and explain relevance.
"""

    parsed, raw = client.generate_json(LLM2_SYSTEM_PROMPT, user_prompt)
    if not parsed:
        return {
            "llm_stage": "LLM-2",
            "dominant_historical_motivation": DEFAULT_MOTIVATION,
            "confidence_level": DEFAULT_CONFIDENCE,
            "explanation": "Fallback: Gemini unavailable or invalid JSON.",
            "raw_response": raw,
        }

    return {
        "llm_stage": "LLM-2",
        "dominant_historical_motivation": parsed.get("dominant_historical_motivation", DEFAULT_MOTIVATION),
        "confidence_level": parsed.get("confidence_level", DEFAULT_CONFIDENCE),
        "explanation": parsed.get("explanation", ""),
        "raw_response": raw,
    }


def llm3_analyze(record: Dict[str, Any], client: GeminiClient) -> Dict[str, Any]:
    """LLM-3 pattern agent."""
    user_prompt = f"""
Crime Context (Preprocessed):

Crime Type: {_safe_str(record.get("crime_type"))}
Location Type: {_safe_str(record.get("location_type"))}
Domestic Case: {_safe_str(record.get("domestic"))}
Arrest Made: {_safe_str(record.get("arrest"))}
District: {_safe_str(record.get("district"))}
Year: {_safe_str(record.get("year"))}

Context Summary:
{_safe_str(record.get("context_text"), "Not provided")}

Analyze the crime pattern based on context.
"""

    parsed, raw = client.generate_json(LLM3_SYSTEM_PROMPT, user_prompt)
    if not parsed:
        return {
            "llm_stage": "LLM-3",
            "identified_pattern": "unclear/general pattern",
            "confidence_level": DEFAULT_CONFIDENCE,
            "explanation": "Fallback: Gemini unavailable or invalid JSON.",
            "raw_response": raw,
        }

    return {
        "llm_stage": "LLM-3",
        "identified_pattern": parsed.get("identified_pattern", "unclear/general pattern"),
        "confidence_level": parsed.get("confidence_level", DEFAULT_CONFIDENCE),
        "explanation": parsed.get("explanation", ""),
        "raw_response": raw,
    }


# ---------------------------------------------------------------------------
# Fusion layer (deterministic)
# ---------------------------------------------------------------------------

def fuse_outputs(llm1: Dict[str, Any], llm2: Dict[str, Any], llm3: Dict[str, Any]) -> Dict[str, Any]:
    scores: Dict[str, float] = {}

    def add_score(motivation: Optional[str], weight: float) -> None:
        if not motivation:
            return
        motivation = motivation.lower()
        if motivation == "error":
            return
        scores[motivation] = scores.get(motivation, 0.0) + weight

    add_score(llm1.get("predicted_motivation"), _confidence_to_weight(llm1.get("confidence", DEFAULT_CONFIDENCE)))
    add_score(
        llm2.get("dominant_historical_motivation"),
        _confidence_to_weight(llm2.get("confidence_level", DEFAULT_CONFIDENCE)),
    )
    pattern = llm3.get("identified_pattern")
    pattern_motivation = PATTERN_MOTIVATION_MAP.get(pattern, DEFAULT_MOTIVATION)
    add_score(pattern_motivation, _confidence_to_weight(llm3.get("confidence_level", DEFAULT_CONFIDENCE)))

    if not scores:
        final_motivation = DEFAULT_MOTIVATION
        agreement_score = 0.0
    else:
        final_motivation = max(scores, key=scores.get)
        agreement_score = scores[final_motivation]

    final_confidence = (
        "High" if agreement_score >= 1.8 else "Medium" if agreement_score >= 1.2 else "Low"
    )

    return {
        "final_motivation": final_motivation,
        "agreement_score": round(agreement_score, 2),
        "final_confidence": final_confidence,
        "supporting_models": {
            "LLM-1": llm1.get("predicted_motivation"),
            "LLM-2": llm2.get("dominant_historical_motivation"),
            "LLM-3_pattern": pattern,
        },
    }


# ---------------------------------------------------------------------------
# LLM-4 report
# ---------------------------------------------------------------------------

def llm4_report(fusion_output: Dict[str, Any], crime_details: Dict[str, Any], client: GeminiClient) -> str:
    user_prompt = f"""
FUSED MODEL OUTPUT:

Final Motivation: {fusion_output.get('final_motivation', DEFAULT_MOTIVATION)}
Confidence Level: {fusion_output.get('final_confidence', DEFAULT_CONFIDENCE)}
Agreement Score: {fusion_output.get('agreement_score', 0)}

Supporting Evidence:
- LLM-1 Prediction: {fusion_output.get('supporting_models', {}).get('LLM-1')}
- LLM-2 Historical Trend: {fusion_output.get('supporting_models', {}).get('LLM-2')}
- LLM-3 Context Pattern: {fusion_output.get('supporting_models', {}).get('LLM-3_pattern')}

Original Crime Details:
Crime Description: {_safe_str(crime_details.get('crime_text'), 'N/A')}
Area: {_safe_str(crime_details.get('area_name'), 'N/A')}
Crime Type: {_safe_str(crime_details.get('crime_type'), 'N/A')}
Victim Age: {_safe_str(crime_details.get('vict_age'), 'N/A')}
Victim Sex: {_safe_str(crime_details.get('vict_sex'), 'N/A')}
Weapon: {_safe_str(crime_details.get('weapon_desc'), 'N/A')}
Status: {_safe_str(crime_details.get('status_desc'), 'N/A')}
Location Type: {_safe_str(crime_details.get('premis_desc'), 'N/A')}

Generate a professional crime analysis report based on the above information.
"""

    return client.generate_text(LLM4_SYSTEM_PROMPT, user_prompt)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _normalize_input(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected keys exist with safe defaults."""
    defaults = {
        "crime_text": "",
        "crime_type": "Unknown",
        "weapon_desc": "Unknown",
        "premis_desc": "Unknown",
        "vict_age": "Unknown",
        "vict_sex": "Unknown",
        "area_name": "Unknown",
        "domestic": "Unknown",
        "status_desc": "Unknown",
        "arrest": "Unknown",
        "state": None,
        "year": None,
        "district": "Unknown",
        "location_type": "Unknown",
        "context_text": None,
    }
    normalized = {**defaults, **(record or {})}
    return normalized


def _build_llm3_record(normalized: Dict[str, Any]) -> Dict[str, Any]:
    if normalized.get("context_text"):
        context_text = normalized["context_text"]
    else:
        context_text = (
            f"In {_safe_str(normalized.get('year'))}, a {_safe_str(normalized.get('crime_type'))} incident "
            f"occurred at a {_safe_str(normalized.get('premis_desc'))} location. Domestic case: "
            f"{_safe_str(normalized.get('domestic'))}. Arrest made: {_safe_str(normalized.get('arrest'))}. "
            f"District {_safe_str(normalized.get('district'))}."
        )

    return {
        "context_text": context_text,
        "crime_type": normalized.get("crime_type"),
        "location_type": normalized.get("location_type", normalized.get("premis_desc")),
        "domestic": normalized.get("domestic"),
        "arrest": normalized.get("arrest"),
        "district": normalized.get("district"),
        "year": normalized.get("year"),
    }


def run_full_pipeline(input_record: Dict[str, Any]) -> Dict[str, Any]:
    """Unified pipeline that orchestrates LLM-1, LLM-2, LLM-3, fusion, and LLM-4."""
    normalized = _normalize_input(input_record)
    gemini_client = GeminiClient()

    llm1_out = llm1_analyze(normalized, gemini_client)

    llm2_context = {
        "state": normalized.get("state"),
        "year": normalized.get("year"),
        "crime_type": normalized.get("crime_type"),
    }
    llm2_out = llm2_analyze(llm2_context, gemini_client)

    llm3_record = _build_llm3_record(normalized)
    llm3_out = llm3_analyze(llm3_record, gemini_client)

    fusion_out = fuse_outputs(llm1_out, llm2_out, llm3_out)

    report = llm4_report(fusion_out, normalized, gemini_client)

    return {
        "llm1": llm1_out,
        "llm2": llm2_out,
        "llm3": llm3_out,
        "fusion": fusion_out,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _sample_inputs() -> List[Dict[str, Any]]:
    """Generate 5–10 diverse test cases (full, partial, minimal)."""
    return [
        {
            "crime_text": "On 2020-05-10 at 22 hours, in central area, a 21-year-old M was involved in robbery at street. Weapon used: verbal threat. Case status: invest cont.",
            "crime_type": "robbery",
            "weapon_desc": "verbal threat",
            "premis_desc": "street",
            "vict_age": "21",
            "vict_sex": "M",
            "area_name": "central",
            "domestic": "false",
            "status_desc": "invest cont",
            "arrest": "false",
            "state": "ANDHRA PRADESH",
            "year": 2020,
            "district": "UNKNOWN",
        },
        {
            # Partial data
            "crime_text": "Incident at a residence involving family dispute; no weapon reported.",
            "crime_type": "assault",
            "premis_desc": "residence",
            "domestic": "true",
            "state": "KARNATAKA",
            "year": 2018,
        },
        {
            # Minimal data
            "crime_text": "A theft occurred.",
            "crime_type": "theft",
        },
        {
            # Different crime type
            "crime_text": "Armed attack in public place; firearm discharged, arrest made on scene.",
            "crime_type": "aggravated assault",
            "weapon_desc": "firearm",
            "premis_desc": "public place",
            "arrest": "true",
            "state": "DELHI",
            "year": 2015,
        },
        {
            # Sexual crime context with missing year
            "crime_text": "Reported sexual assault in park; suspect unknown; victim female adult.",
            "crime_type": "sexual assault",
            "premis_desc": "park",
            "vict_sex": "F",
            "state": "MAHARASHTRA",
        },
        {
            # Domestic with arrest, strong pattern
            "crime_text": "Domestic violence case, repeated incidents, police intervened and arrested suspect.",
            "crime_type": "domestic violence",
            "domestic": "true",
            "arrest": "true",
            "premis_desc": "home",
            "year": 2022,
            "state": "TELANGANA",
        },
    ]


def run_tests() -> None:
    tests = _sample_inputs()
    for idx, rec in enumerate(tests, start=1):
        print(f"\n=== Test Case {idx} ===")
        result = run_full_pipeline(rec)
        print("LLM-1:", result["llm1"])
        print("LLM-2:", result["llm2"])
        print("LLM-3:", result["llm3"])
        print("Fusion:", result["fusion"])
        print("Report:\n", result["report"])


if __name__ == "__main__":
    print("Running pipeline smoke tests (Gemini key required for live responses)...")
    run_tests()
