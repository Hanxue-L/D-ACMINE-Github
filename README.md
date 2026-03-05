# D-ACMINE Analysis Scripts

Core scripts for DPV feature extraction and ML evaluation for the D-ACMINE manuscript.

## Files

*   `dpv_feature_extraction.py`: Extracts DPV features using valley-to-valley baseline correction.
*   `ml_training_evaluation.py`: Runs nested cross-validation for ML models.
*   `requirements.txt`: Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Configure `I_BLANK` constants in `dpv_feature_extraction.py` and prepare data in `data/`, then run:

```bash
python dpv_feature_extraction.py
```

With `extracted_features.json` and `golden_pairs.json` ready, run:

```bash
python ml_training_evaluation.py
```
