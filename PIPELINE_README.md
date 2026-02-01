# Crime Motivation Analysis Pipeline v2.0

A complete end-to-end pipeline for analyzing criminal motivations using machine learning and AI models.

## 🎯 Overview

This pipeline analyzes crime cases through 4 specialized stages:

1. **LLM-1**: Motivation Classification
   - Analyzes crime description
   - Predicts primary motivation (emotional, financial, power, sexual, unknown)
   - Provides crime indicators

2. **LLM-2**: Historical Context Analysis
   - Examines historical/regional crime patterns
   - Identifies dominant motivation trends
   - Assesses data quality (Exact/Partial/Estimated)

3. **LLM-3**: Behavioral Pattern Identification
   - Classifies crime patterns (domestic violence, street crime, organized, etc.)
   - Identifies pattern indicators
   - Provides behavioral assessment

4. **Fusion Layer**: Multi-Model Consensus
   - Combines all 3 predictions
   - Detects model conflicts
   - Weighted voting with confidence calibration
   - Data quality boosting

5. **LLM-4**: Comprehensive Report Generation
   - Synthesizes all analyses
   - Generates structured 7-section report
   - Provides actionable insights

## 📦 Installation

### Requirements

```bash
pip install google-generativeai sentence-transformers scikit-learn pandas numpy
```

### Files Needed

```
/content/
├── llm1_model.pkl              # Pre-trained motivation classifier
├── llm2_model.pkl              # Pre-trained historical analyzer
├── llm3_model.pkl              # Pre-trained pattern classifier
├── llm2_motivation_dataset.json # Historical motivation data
└── llm3_clustered_data.json    # Behavioral pattern data
```

## 🚀 Quick Start

### Single Case Analysis

```python
from crime_analysis_pipeline import analyze_crime_case, print_case_summary, export_analysis_to_json

# Define a crime case
crime = {
    "crime_text": "On 2020-05-10 at 22 hours, in central area, a 21-year-old M was involved in robbery...",
    "crm_cd_desc": "robbery",
    "area_name": "central",
    "premis_desc": "street",
    "vict_age": "21",
    "vict_sex": "M",
    "weapon_desc": "handgun",
    "status_desc": "invest cont",
    "domestic": "false"
}

# Run analysis
result = analyze_crime_case(crime, verbose=True)

# View summary
print_case_summary(result)

# Export results
export_analysis_to_json(result, "crime_case.json")
```

### Batch Analysis

```python
from crime_analysis_pipeline import analyze_batch_crimes

# Analyze multiple crimes
crimes = [crime1, crime2, crime3, ...]
batch_results = analyze_batch_crimes(crimes)

# Access metrics
print(batch_results['metrics'])
```

### Run Demo

```bash
python crime_analysis_pipeline.py
```

## 📊 Output Structure

### Single Case Analysis

```json
{
  "case_id": "CASE-12345",
  "timestamp": "2024-01-31T10:30:00",
  "crime_details": { ... },
  "stage_1_motivation_analysis": {
    "predicted_motivation": "financial",
    "confidence": "High",
    "reasoning": "...",
    "crime_indicators": [...]
  },
  "stage_2_historical_analysis": {
    "dominant_historical_motivation": "financial",
    "confidence_level": "Medium",
    "data_quality": "Exact"
  },
  "stage_3_pattern_analysis": {
    "identified_pattern": "opportunistic_crime_pattern",
    "pattern_indicators": [...]
  },
  "fusion_layer": {
    "final_motivation": "financial",
    "agreement_score": 2.4,
    "final_confidence": "High",
    "models_agree": true,
    "conflict_detected": false
  },
  "stage_4_report": {
    "report": "Comprehensive analysis report...",
    "status": "success"
  },
  "analysis_summary": {
    "final_motivation": "financial",
    "final_confidence": "High",
    "recommendation": "High confidence - All models aligned"
  }
}
```

## 🔬 Understanding the Output

### Motivation Types

- **emotional**: Personal conflicts, domestic disputes, revenge, rage, jealousy
- **financial**: Theft, robbery, fraud, extortion, monetary gain
- **power**: Assault, murder, control, dominance, territorial disputes
- **sexual**: Sexual assault, exploitation, predatory behavior
- **unknown**: Insufficient information

### Confidence Levels

- **High** (≥1.8 score): All models aligned, high reliability
- **Medium** (1.2-1.8): Two models aligned, moderate reliability
- **Low** (<1.2 or conflict): Requires expert review

### Data Quality (LLM-2)

- **Exact**: Historical data perfectly matched
- **Partial**: Some historical data found
- **Estimated**: No exact match, generalized trends used

## 🛠️ Configuration

Edit `crime_analysis_pipeline.py`:

```python
# API Key
GEMINI_API_KEY = "your-key-here"

# Model directory
MODEL_DIR = "/content/"  # or "/path/to/models/"

# Output directory
OUTPUT_DIR = "./analysis_results/"
```

## 📈 Performance Metrics

Pipeline automatically tracks:

- **Motivation Distribution**: Breakdown of predicted motivations
- **Confidence Distribution**: Breakdown of confidence levels
- **Conflict Rate**: % of cases with model disagreement
- **High Confidence Rate**: % of cases with high confidence

Example:

```
Total cases: 100
Motivation distribution: {'financial': 42, 'emotional': 35, 'power': 18, 'sexual': 5}
Confidence distribution: {'High': 65, 'Medium': 28, 'Low': 7}
Conflict rate: 8.5%
High confidence rate: 65.0%
```

## 🔧 Advanced Usage

### Custom LLM Prompts

Modify system prompts in the file to adjust model behavior:

```python
LLM1_SYSTEM_PROMPT = """Your custom prompt here..."""
```

### Load Data from CSV

```python
import pandas as pd

df = pd.read_csv("crime_data.csv")
crimes = df[['crm_cd_desc', 'area_name', ...]].to_dict('records')
batch = analyze_batch_crimes(crimes)
```

### Validate Against Ground Truth

```python
from crime_analysis_pipeline import validate_predictions

ground_truth = {"CASE-1": "financial", "CASE-2": "emotional"}
metrics = validate_predictions_with_ground_truth(results, ground_truth)
# Returns: accuracy, precision, recall, f1_score, confusion_matrix
```

## ⚠️ Error Handling

The pipeline has built-in error handling:

- JSON parsing fallback (handles markdown-wrapped responses)
- Retry logic with exponential backoff (3 attempts)
- Graceful degradation (returns "unknown" if analysis fails)
- Detailed error messages in output

## 📝 Troubleshooting

### "Could not find model file"

**Solution**: Ensure all `.pkl` and `.json` files are in `/content/` directory

### "Could not extract valid JSON"

**Solution**: Usually transient - retry the analysis. Check internet connection.

### "API rate limit exceeded"

**Solution**: Add delays between API calls or use batch processing

## 📚 References

### Model Accuracy

- **LLM-1**: ~85-92% accuracy (from training metrics)
- **LLM-2**: Depends on historical data availability
- **LLM-3**: ~90%+ pattern classification accuracy

### Citation

```bibtex
@software{crime_pipeline_2024,
  title={Crime Motivation Analysis Pipeline v2.0},
  year={2024},
  note={Uses Google Gemini API, Sentence-Transformers, scikit-learn}
}
```

## 📄 License

This project is provided as-is for research and educational purposes.

## ✉️ Support

For issues or questions, check:
1. Model files exist and are readable
2. API key is valid and has quota
3. Internet connection is stable
4. Python dependencies are installed

---

**Last Updated**: January 31, 2024  
**Version**: 2.0
