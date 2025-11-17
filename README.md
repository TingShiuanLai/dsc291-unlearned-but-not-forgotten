# Unlearned but Not Forgotten

Implementation of machine unlearning experiments on medical synthetic data and WMDP-Bio corpus.

## Project Structure

```
├── code/                           # All source code (organized by functionality)
│   ├── generate/                  # Data generation scripts
│   │   └── generate_medical_synthetic.py
│   ├── preprocess/                # Data preprocessing and splitting
│   │   ├── prepare_medical_synthetic_dataset.py
│   │   └── prepare_wmdp_bio_dataset.py
│   └── train/                     # Training and fine-tuning
│       └── finetune_hf_lora.py
├── dataset/                       # All datasets (git-ignored)
│   ├── synthetic_soap_notes.jsonl
│   ├── med_synthetic_full.json
│   ├── med_synthetic_forget10.json
│   ├── med_synthetic_full_minus_forget10.json
│   └── wmdp_bio_*.json
├── models/                        # Trained model checkpoints (git-ignored)
│   └── llama-3.1-8b-*/
└── outputs/                       # Logs and evaluation results (git-ignored)
    └── logs/
```

## Quick Start

### 1. Generate Synthetic Medical Data

Generate SOAP notes into `dataset/`:

```bash
python code/generate/generate_medical_synthetic.py \
  --n_record 1000 \
  --output dataset/synthetic_soap_notes.jsonl \
  --api_key YOUR_TINKER_API_KEY
```

### 2. Prepare Medical Synthetic Dataset

Process and create forget/retain splits (dry-run first to preview):

```bash
# Dry run (no files written)
python code/preprocess/prepare_medical_synthetic_dataset.py \
  --input-file dataset/synthetic_soap_notes.jsonl \
  --output-dir dataset \
  --dry-run \
  --sample 5

# Actually create splits
python code/preprocess/prepare_medical_synthetic_dataset.py \
  --input-file dataset/synthetic_soap_notes.jsonl \
  --output-dir dataset
```

This creates:
-- `dataset/med_synthetic_full.json` (full dataset with unique client names)
-- `dataset/med_synthetic_forget10.json` (10% forget set)
-- `dataset/med_synthetic_full_minus_forget10.json` (90% retain set)
- Similar files for 1%, 5%, 20% forget ratios

### 3. Prepare WMDP-Bio Dataset

Download and process WMDP-Bio corpus:

```bash
python code/preprocess/prepare_wmdp_bio_dataset.py \
  --target_size 5300 \
  --output_dir dataset
```

### 4. Fine-tune Model (Oracle)

Train on full dataset:

```bash
python code/train/finetune_hf_lora.py \
  --data_file dataset/med_synthetic_full.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir models/llama-3.1-8b-medical-oracle \
  --num_epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --hf_token YOUR_HF_TOKEN
```

## Script Defaults

All scripts use sensible defaults for the structured layout:

- **Generate**: Outputs to `dataset/synthetic_soap_notes.jsonl`
- **Preprocess (Medical)**: Reads from `dataset/`, writes to `dataset/`
- **Preprocess (WMDP)**: Downloads and writes to `dataset/`
- **Train**: Reads from `dataset/`, writes models to `models/`

## Command Reference

### Generate Medical Synthetic Data
```bash
python code/generate/generate_medical_synthetic.py \
  [--n_record 1000] \
  [--output dataset/synthetic_soap_notes.jsonl] \
  [--model meta-llama/Llama-3.1-8B-Instruct] \
  [--api_key YOUR_KEY]
```

### Prepare Medical Dataset
```bash
python code/preprocess/prepare_medical_synthetic_dataset.py \
  [-i dataset/synthetic_soap_notes.jsonl] \
  [-o dataset] \
  [--seed 42] \
  [--dry-run] \
  [--sample 3] \
  [--forget-ratios 0.10 0.01 0.05 0.20]
```

### Prepare WMDP Dataset
```bash
python code/preprocess/prepare_wmdp_bio_dataset.py \
  [--target_size 5300] \
  [--output_dir dataset] \
  [--seed 42] \
  [--forget_ratios 0.01 0.05 0.10 0.20]
```

### Fine-tune Model
```bash
python code/train/finetune_hf_lora.py \
  [--data_file dataset/med_synthetic_full.json] \
  [--base_model meta-llama/Llama-3.1-8B-Instruct] \
  [--output_dir models/llama-3.1-8b-medical-oracle] \
  [--num_epochs 3] \
  [--batch_size 1] \
  [--learning_rate 2e-5] \
  [--lora_rank 8] \
  [--lora_alpha 16] \
  [--max_seq_length 256] \
  [--use_8bit] \
  [--hf_token YOUR_TOKEN]
```

## Data Format

### Medical SOAP Notes (JSONL)
```json
{
  "client_name": "John Doe",
  "date_of_birth": "1970-01-15",
  "date": "2024-03-20",
  "subjective": "Patient presents with...",
  "objective": "Vital signs: BP 120/80...",
  "assessment": "Likely diagnosis...",
  "plan": "Treatment plan..."
}
```

### WMDP-Bio (JSON)
```json
{
  "text": "Sentence from PubMed paper...",
  "source": "wmdp-bio-retain",
  "type": "paper_sentence",
  "paper_index": 0,
  "sentence_index": 0
}
```

## Notes

- All data files (raw and processed) are git-ignored by default
- Model checkpoints are git-ignored (large binary files)
- Use `--dry-run` flag with preprocessing scripts to preview outputs before writing
- The `--hf_token` parameter is required for gated models like Llama

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers` - Hugging Face Transformers
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `datasets` - Hugging Face Datasets (for WMDP corpus)
- `torch` - PyTorch
- `tinker` - Tinker SDK (for data generation)
