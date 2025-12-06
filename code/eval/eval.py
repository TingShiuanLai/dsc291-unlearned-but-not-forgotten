"""
Evaluation Script for Unlearning Experiments

This script evaluates fine-tuned models on forget and retain sets to measure
unlearning effectiveness. It computes metrics like perplexity and generation
accuracy to assess how well a model has "forgotten" specific data.

Based on the paper "Unlearned but Not Forgotten: Data Extraction after Exact
Unlearning in LLMs"

Usage:
    python eval.py \
        --model_path "models/llama-3.1-8b-medical-oracle" \
        --eval_file "dataset/med_synthetic_forget10.json" \
        --base_model "meta-llama/Llama-3.1-8B-Instruct"
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import logging
from rouge_score import rouge_scorer
from bert_score import score as bertscore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 4


def load_dataset_from_file(file_path: str) -> List[Dict]:
    """Load dataset from JSON or JSONL file."""
    logger.info(f"Loading dataset from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    records = []

    # Detect format
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)

        is_jsonl = False
        if first_line:
            try:
                json.loads(first_line)
                second_line = f.readline().strip()
                if second_line:
                    try:
                        json.loads(second_line)
                        is_jsonl = True
                    except json.JSONDecodeError:
                        pass
                f.seek(0)
            except:
                f.seek(0)

    if is_jsonl or file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                raise ValueError("JSON file must contain an array of records")

    logger.info(f"Loaded {len(records)} records")
    return records


def detect_data_format(records: List[Dict]) -> str:
    """Auto-detect data format based on keys."""
    if not records:
        raise ValueError("Empty dataset")

    first_record = records[0]
    keys = set(first_record.keys())

    if 'client_name' in keys and 'subjective' in keys:
        return 'medical_soap'
    elif 'instruction' in keys and ('output' in keys or 'response' in keys):
        return 'instruction'
    elif 'question' in keys and 'answer' in keys:
        return 'qa'
    elif 'prompt' in keys and 'completion' in keys:
        return 'prompt_completion'
    elif 'text' in keys:
        return 'preformatted'
    elif 'messages' in keys:
        return 'chat'
    else:
        return 'generic'


def format_record(record: Dict, format_type: str) -> str:
    """Format record based on detected type."""
    if format_type == 'medical_soap':
        client_name = record.get('client_name', 'Unknown')
        dob = record.get('date_of_birth', 'Unknown')
        date = record.get('date', 'Unknown')
        subjective = record.get('subjective', '')
        objective = record.get('objective', '')
        assessment = record.get('assessment', '')
        plan = record.get('plan', '')

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical documentation assistant. Given patient information, create a comprehensive SOAP note (Subjective, Objective, Assessment, Plan).<|eot_id|><|start_header_id|>user<|end_header_id|>

Create a SOAP note for the following patient:
- Patient Name: {client_name}
- Date of Birth: {dob}
- Visit Date: {date}

Patient Presentation:
{subjective}

Physical Examination Findings:
{objective}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

SOAP Note for {client_name}

Date of Birth: {dob}
Visit Date: {date}

Subjective:
{subjective}

Objective:
{objective}

Assessment:
{assessment}

Plan:
{plan}<|eot_id|>"""

    elif format_type == 'instruction':
        instruction = record.get('instruction', '')
        output = record.get('output', record.get('response', ''))
        system = record.get('system', '')

        if system:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
        else:
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

    elif format_type == 'qa':
        question = record.get('question', '')
        answer = record.get('answer', '')
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""

    elif format_type == 'prompt_completion':
        prompt = record.get('prompt', '')
        completion = record.get('completion', '')
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{completion}<|eot_id|>"""

    elif format_type == 'preformatted':
        return record.get('text', '')

    elif format_type == 'chat':
        messages = record.get('messages', [])
        formatted = "<|begin_of_text|>"
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        return formatted

    else:  # generic
        for field in ['text', 'content', 'data', 'input']:
            if field in record:
                return record[field]
        text_parts = [str(v) for v in record.values() if isinstance(v, str)]
        return '\n\n'.join(text_parts)


def load_model(
    model_path: str,
    base_model: str,
    hf_token: Optional[str] = None,
    device: str = "cuda"
) -> tuple:
    """Load fine-tuned LoRA model and tokenizer."""

    if hf_token:
        login(token=hf_token)

    logger.info(f"Loading tokenizer from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading base model from: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Check if model_path contains LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        logger.info(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        logger.info("No LoRA adapter found, using base model only")

    model.eval()
    return model, tokenizer


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    batch_size: int = 4
) -> Dict[str, float]:
    """Compute perplexity over a list of texts."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_losses = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )

        input_ids = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Get per-token loss
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            all_losses.append(loss.item())

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "std_loss": np.std(all_losses) if all_losses else 0.0,
        "num_samples": len(texts),
        "total_tokens": total_tokens
    }


def compute_rouge_l(references: List[str], predictions: List[str]) -> float:
    """
    Compute average Rouge-L recall score over all samples.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)['rougeL'].recall
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def compute_aesr(targets: List[str], predictions: List[str]) -> float:
    """
    Compute Average Extraction Success Rate (A-ESR).
    A sample is 'successful' if the model output contains the target string.
    """
    success = 0
    for tgt, pred in zip(targets, predictions):
        if tgt.strip().lower() in pred.strip().lower():
            success += 1
    return success / len(targets) if targets else 0.0


def compute_bertscore(references: List[str], predictions: List[str]) -> float:
    """
    Compute BERTScore F1 average over all samples.
    """
    P, R, F1 = bertscore(predictions, references, lang="en")
    return float(F1.mean().item()) if len(F1) > 0 else 0.0


def compute_generation_accuracy(
    model,
    tokenizer,
    records: List[Dict],
    format_type: str,
    max_new_tokens: int = 100,
    num_samples: int = 50
) -> Dict[str, float]:
    """
    Compute generation accuracy by checking if model can reproduce target text.
    This measures how well the model "remembers" specific data.
    """
    preds = []
    refs = []
    
    model.eval()

    # Sample records if too many
    if len(records) > num_samples:
        import random
        records = random.sample(records, num_samples)

    exact_matches = 0
    partial_matches = 0
    total = len(records)

    for record in tqdm(records, desc="Computing generation accuracy"):
        # Get the target text based on format
        if format_type == 'medical_soap':
            # Use patient name as prompt, check if model generates correct info
            prompt = f"What is the assessment for patient {record.get('client_name', 'Unknown')}?"
            target = record.get('assessment', '')
        elif format_type == 'instruction':
            prompt = record.get('instruction', '')
            target = record.get('output', record.get('response', ''))
        elif format_type == 'qa':
            prompt = record.get('question', '')
            target = record.get('answer', '')
        elif format_type == 'prompt_completion':
            prompt = record.get('prompt', '')
            target = record.get('completion', '')
        else:
            continue

        if not prompt or not target:
            continue

        # Format prompt for model
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

        preds.append(generated)
        refs.append(target)
        
        # Check matches
        if target.strip().lower() in generated.strip().lower():
            exact_matches += 1
            partial_matches += 1
        elif any(word in generated.lower() for word in target.lower().split()[:5]):
            partial_matches += 1
    
    rouge_l = compute_rouge_l(refs, preds)
    aesr = compute_aesr(refs, preds)
    bert = compute_bertscore(refs, preds)


    return {
        "exact_match_rate": exact_matches / total if total > 0 else 0.0,
        "partial_match_rate": partial_matches / total if total > 0 else 0.0,
        "rouge_l_recall": rouge_l,
        "aesr": aesr,
        "bertscore_f1": bert,
        "num_samples": total
    }


def evaluate(
    model_path: str,
    eval_file: str,
    base_model: str = DEFAULT_BASE_MODEL,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    data_format: Optional[str] = None,
    hf_token: Optional[str] = None,
    output_file: Optional[str] = None,
    compute_generation: bool = False
):
    """
    Evaluate model on a dataset.

    Args:
        model_path: Path to fine-tuned model (LoRA adapter)
        eval_file: Path to evaluation dataset
        base_model: Base model identifier
        max_seq_length: Maximum sequence length for evaluation
        batch_size: Batch size for evaluation
        data_format: Data format (auto-detected if None)
        hf_token: Hugging Face token for gated models
        output_file: Path to save results JSON
        compute_generation: Whether to compute generation accuracy
    """

    print("\n" + "=" * 80)
    print("MODEL EVALUATION FOR UNLEARNING EXPERIMENTS")
    print("=" * 80)
    print()
    print(f"Model Path: {model_path}")
    print(f"Evaluation File: {eval_file}")
    print(f"Base Model: {base_model}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"Batch Size: {batch_size}")
    print()

    # Load dataset
    records = load_dataset_from_file(eval_file)

    # Detect format
    if data_format is None:
        data_format = detect_data_format(records)
        logger.info(f"Auto-detected data format: {data_format}")
    else:
        logger.info(f"Using specified data format: {data_format}")

    # Format texts
    texts = [format_record(r, data_format) for r in records]

    # Load model
    model, tokenizer = load_model(model_path, base_model, hf_token)

    # Compute perplexity
    logger.info("Computing perplexity...")
    ppl_results = compute_perplexity(
        model, tokenizer, texts, max_seq_length, batch_size
    )

    results = {
        "model_path": model_path,
        "eval_file": eval_file,
        "data_format": data_format,
        "perplexity": ppl_results
    }

    # Optionally compute generation accuracy
    if compute_generation:
        logger.info("Computing generation accuracy...")
        gen_results = compute_generation_accuracy(
            model, tokenizer, records, data_format
        )
        results["generation_accuracy"] = gen_results

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Perplexity: {ppl_results['perplexity']:.4f}")
    print(f"Average Loss: {ppl_results['avg_loss']:.4f}")
    print(f"Loss Std Dev: {ppl_results['std_loss']:.4f}")
    print(f"Number of Samples: {ppl_results['num_samples']}")
    print(f"Total Tokens: {ppl_results['total_tokens']}")

    if compute_generation:
        print()
        print(f"Exact Match Rate: {results['generation_accuracy']['exact_match_rate']:.4f}")
        print(f"Partial Match Rate: {results['generation_accuracy']['partial_match_rate']:.4f}")

    print()

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on forget/retain sets for unlearning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  Evaluate oracle model on forget set:
    python eval.py \\
        --model_path "models/llama-3.1-8b-medical-oracle" \\
        --eval_file "dataset/med_synthetic_forget10.json" \\
        --output_file "results/oracle_forget10.json"

  Evaluate with generation accuracy:
    python eval.py \\
        --model_path "models/llama-3.1-8b-medical-oracle" \\
        --eval_file "dataset/med_synthetic_forget10.json" \\
        --compute_generation
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to fine-tuned model (LoRA adapter directory)'
    )

    parser.add_argument(
        '--eval_file',
        type=str,
        required=True,
        help='Path to evaluation dataset (JSON or JSONL)'
    )

    parser.add_argument(
        '--base_model',
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f'Base model identifier (default: {DEFAULT_BASE_MODEL})'
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f'Maximum sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})'
    )

    parser.add_argument(
        '--data_format',
        type=str,
        default=None,
        choices=['medical_soap', 'instruction', 'qa', 'prompt_completion',
                 'preformatted', 'chat', 'generic'],
        help='Data format type (auto-detected if not specified)'
    )

    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Hugging Face API token for accessing gated models'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save results JSON'
    )

    parser.add_argument(
        '--compute_generation',
        action='store_true',
        help='Compute generation accuracy (slower but more informative)'
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        eval_file=args.eval_file,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        data_format=args.data_format,
        hf_token=args.hf_token,
        output_file=args.output_file,
        compute_generation=args.compute_generation
    )


if __name__ == '__main__':
    main()