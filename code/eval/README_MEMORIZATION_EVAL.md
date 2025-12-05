# Baseline Memorization Evaluation with Tinker API

This script evaluates fine-tuned models using the Tinker API to measure memorization through greedy decoding.

## Overview

The evaluation script `eval_memorization_tinker.py` measures:

1. **Perplexity** - As a sanity check for overall model quality
2. **Rouge-L Similarity** - Between model-generated continuations and ground truth data

## Evaluation Methodology

### For WMDP Dataset
- **Prompt**: First half of the sentence
- **Target**: Second half of the sentence
- **Metric**: Rouge-L similarity between generated continuation and actual second half

### For Medical Synthetic Notes
- **Prompt**: Client name, date of birth, and date of note
- **Target**: Full SOAP note (Subjective, Objective, Assessment, Plan)
- **Metric**: Rouge-L similarity between generated SOAP note and actual note

### Generation Method
- **Greedy Decoding**: Uses temperature=0.0 for deterministic generation
- This represents the baseline for memorization extraction

## Installation

Ensure you have the required dependencies:

```bash
pip install tinker rouge-score numpy tqdm python-dotenv
```

## Usage

### Basic Usage

```bash
python code/eval/eval_memorization_tinker.py \
    --checkpoint_dir "checkpoints/llama-3.1-8b-wmdp-retain20" \
    --eval_file "dataset/wmdp_bio_forget20.json" \
    --output_file "results/wmdp_baseline_memorization.json"
```

### Evaluate Medical Checkpoint

```bash
python code/eval/eval_memorization_tinker.py \
    --checkpoint_dir "checkpoints/llama-3.1-8b-medical" \
    --eval_file "dataset/med_synthetic_forget20.json" \
    --output_file "results/medical_baseline_memorization.json"
```

### Evaluate on a Subset

To evaluate on only 100 samples (faster):

```bash
python code/eval/eval_memorization_tinker.py \
    --checkpoint_dir "checkpoints/llama-3.1-8b-wmdp-retain20" \
    --eval_file "dataset/wmdp_bio_forget20.json" \
    --num_samples 100 \
    --output_file "results/wmdp_baseline_memorization_subset.json"
```

### Skip Perplexity Computation

If you only want memorization metrics:

```bash
python code/eval/eval_memorization_tinker.py \
    --checkpoint_dir "checkpoints/llama-3.1-8b-wmdp-retain20" \
    --eval_file "dataset/wmdp_bio_forget20.json" \
    --skip_perplexity \
    --output_file "results/wmdp_memorization_only.json"
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint_dir` | Path to checkpoint directory (required) | - |
| `--eval_file` | Path to evaluation dataset JSON/JSONL (required) | - |
| `--base_model` | Base model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `--max_new_tokens` | Maximum tokens to generate | 256 |
| `--max_seq_length` | Maximum sequence length for perplexity | 2048 |
| `--num_samples` | Number of samples to evaluate | All |
| `--output_file` | Path to save results JSON | None |
| `--base_url` | Tinker API base URL | From environment |
| `--skip_perplexity` | Skip perplexity computation | False |

## Output Format

The script outputs a JSON file with the following structure:

```json
{
  "checkpoint_dir": "checkpoints/llama-3.1-8b-wmdp-retain20",
  "sampling_path": "checkpoints/llama-3.1-8b-wmdp-retain20/final",
  "eval_file": "dataset/wmdp_bio_forget20.json",
  "data_format": "wmdp",
  "base_model": "meta-llama/Llama-3.1-8B-Instruct",
  "num_records": 200,
  "evaluation_date": "2024-11-20T10:30:00",
  "config": {
    "max_new_tokens": 256,
    "max_seq_length": 2048,
    "num_samples": null,
    "temperature": 0.0,
    "sampling_method": "greedy"
  },
  "perplexity": {
    "perplexity": 12.5,
    "avg_loss": 2.53,
    "std_loss": 0.42,
    "num_samples": 200,
    "total_tokens": 45000
  },
  "memorization": {
    "mean_rouge_l": 0.42,
    "median_rouge_l": 0.38,
    "std_rouge_l": 0.25,
    "min_rouge_l": 0.05,
    "max_rouge_l": 0.95,
    "exact_match_rate": 0.08,
    "num_samples": 200,
    "examples": [
      {
        "prompt": "...half of sentence",
        "target": "second half of sentence...",
        "generated": "model's generated continuation...",
        "rouge_l": 0.45
      }
    ]
  }
}
```

## Interpreting Results

### Perplexity
- **Lower is better** - Indicates better model fit to the data
- Typically ranges from 5-50 for fine-tuned models
- Very low perplexity (<5) on specific data may indicate overfitting/memorization

### Rouge-L Scores
- **Higher is better** - Ranges from 0 to 1
- **0.0-0.3**: Low memorization - Model generates different content
- **0.3-0.6**: Moderate memorization - Some overlap with original data
- **0.6-1.0**: High memorization - Model reproduces original data closely

### Exact Match Rate
- Percentage of samples where generated text exactly matches target
- Higher values indicate stronger memorization

## Expected Workflow

1. **Train model** using `finetune_tinker_lora.py`
2. **Evaluate on forget set** to measure how much the model remembers data it should "forget"
3. **Evaluate on retain set** to ensure model still performs well on data it should remember
4. **Compare results** between different unlearning methods

## Example Batch Evaluation

To evaluate multiple checkpoints:

```bash
# Evaluate on forget set
python code/eval/eval_memorization_tinker.py \
    --checkpoint_dir "checkpoints/llama-3.1-8b-wmdp-retain20" \
    --eval_file "dataset/wmdp_bio_forget20.json" \
    --output_file "results/wmdp_retain20_on_forget20.json"

# Evaluate on retain set
python code/eval/eval_memorization_tinker.py \
    --checkpoint_dir "checkpoints/llama-3.1-8b-wmdp-retain20" \
    --eval_file "dataset/wmdp_bio_retain20.json" \
    --output_file "results/wmdp_retain20_on_retain20.json"

# Evaluate on full dataset
python code/eval/eval_memorization_tinker.py \
    --checkpoint_dir "checkpoints/llama-3.1-8b-wmdp-retain20" \
    --eval_file "dataset/wmdp_bio_full.json" \
    --output_file "results/wmdp_retain20_on_full.json"
```

## Troubleshooting

### "No sampling checkpoint found"
Ensure your checkpoint directory contains a `checkpoints.jsonl` file and a `final/` directory with model weights.

### "Could not load tokenizer"
Make sure `tinker` and `tinker_cookbook` packages are properly installed.

### Out of Memory
- Reduce `--num_samples` to evaluate on fewer examples
- Reduce `--max_new_tokens` to generate shorter sequences
- Reduce `--max_seq_length` for perplexity computation

## Notes

- The script uses asynchronous operations for efficiency
- Greedy decoding (temperature=0.0) is used for reproducibility
- The first 5 generation examples are saved in the output for manual inspection
- Both WMDP and medical dataset formats are automatically detected
