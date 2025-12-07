# Evaluation Scripts for Unlearning Experiments

This directory contains evaluation scripts for measuring the quality of model outputs in unlearning experiments, based on the paper "Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLMs".

## Scripts Overview

### 1. `evaluation.py`
Fast evaluation without model loading. Computes Rouge and BLEU scores only.

**Usage:**
```bash
python evaluation.py \
    --input_files med_forget01_baseline-2.json med_forget01_guided-2.json \
    --output_file results/simple_eval.json
```

**Metrics Computed:**
- Rouge-1, Rouge-2, Rouge-L (Recall, Precision, F-measure)
- BLEU score with n-gram precisions

**Metrics Computed:**
- Rouge-1, Rouge-2, Rouge-L (Recall)
- BLEU score
- Perplexity

## Metrics Explained

### Rouge-L (Recall)
- Measures longest common subsequence between generated and ground truth
- Recall: What fraction of ground truth is captured in generated text
- Higher is better (range: 0-1)
- Most relevant for this unlearning task

### BLEU Score
- Measures n-gram overlap between generated and ground truth
- Standard MT evaluation metric
- Higher is better (range: 0-1)

### Perplexity
- Measures how "surprised" the model is by the text
- Lower perplexity = model is more confident
- Higher perplexity = model is more uncertain
- Useful for detecting unlearning (higher PPL on forgotten data)

## Input File Format

The scripts expect JSON files with the following structure:

```json
{
  "config": {
    "oracle_model_path": "models/oracle_merged",
    "gamma": 1.6,
    ...
  },
  "num_examples": 10,
  "results": [
    {
      "idx": 0,
      "prefix": "Patient: Robert White",
      "ground_truth": "SOAP Note for Robert White...",
      "generated": "SOAP Note for Robert White...",
      "data_format": "medical_soap"
    },
    ...
  ]
}
```

## Output Format

The evaluation scripts produce detailed JSON output:

```json
{
  "baseline-2": {
    "input_file": "med_forget01_baseline-2.json",
    "num_examples": 10,
    "metrics": {
      "rouge": {
        "rougeL": {
          "recall": {
            "mean": 0.8234,
            "std": 0.0456,
            "scores": [...]
          }
        }
      },
      "bleu": {
        "bleu": 0.7123
      },
      "perplexity_predictions": {
        "mean": 12.45,
        "std": 3.21
      }
    }
  },
  "guided-2": { ... }
}
```

## Examples

### Quick Comparison
```bash
cd code/eval
python evaluation.py
    --input_files "..\..\med_forget01_baseline-2.json" "..\..\med_forget01_guided-2.json" --output_file "..\..\results\eval_med_forget01.json"
```

## Understanding Results

### High Rouge-L + High BLEU
- Model generates text very similar to ground truth
- Good memorization/retention
- Expected for baseline models on retain set

### Low Rouge-L + Low BLEU
- Model generates different text from ground truth
- Could indicate successful unlearning
- Expected for unlearned models on forget set

### High Perplexity
- Model is uncertain about the text
- Could indicate successful unlearning (model "forgot")
- Expected for unlearned models on forget set

### Low Perplexity
- Model is confident about the text
- Indicates strong knowledge/memorization
- Expected for baseline models on retain set

Positive values indicate the second method (guided) outperforms the first (baseline).
