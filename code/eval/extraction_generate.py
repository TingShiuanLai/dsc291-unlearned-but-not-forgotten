"""
Extraction Attack Generation Script for Llama 3.1 8B Instruct

Performs the guided/contrasting extraction attack from "Unlearned but Not 
Forgotten" on models fine-tuned with Tinker.

Supported data formats (auto-detected):
- medical_soap: SOAP notes with client_name, subjective, objective, assessment, plan
- preformatted: Raw text with 'text' field (e.g., WMDP)

Usage:
    python extraction_generate.py \
        --oracle-model models/merged \
        --dataset dataset/med_synthetic_full.json \
        --output outputs/results.json \
        --gamma 1.0
"""

import argparse
import json
import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    oracle_model_path: str
    unlearned_model_path: Optional[str]
    dataset_path: str
    output_path: str
    gamma: float = 1.0  # Guidance weight (w in paper)
    max_new_tokens: int = 512
    max_length: int = 3000
    split_ratio: float = 0.5
    do_sample: bool = False
    minus_value: Optional[float] = None  # Token filter threshold in log space (paper default: ~11.5 for γ=1e-5)
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class ExtractionResult:
    idx: int
    prefix: str
    ground_truth: str
    generated: str
    data_format: str
    prefix_tokens: int
    gt_tokens: int
    gen_tokens: int


# ============================================================================
# Data Format Detection and Formatting (matches finetune_tinker_lora.py)
# ============================================================================

def detect_data_format(records: List[Dict]) -> str:
    if not records:
        raise ValueError("Empty dataset")
    keys = set(records[0].keys())
    if 'client_name' in keys and 'subjective' in keys:
        return 'medical_soap'
    elif 'text' in keys:
        return 'preformatted'
    else:
        raise ValueError(f"Unknown data format. Keys: {keys}")


def format_medical_soap_prefix(record: Dict) -> Tuple[str, str]:
    """
    Match exact training format from finetune_tinker_lora.py.
    
    Prefix: System + User message (with patient info, S, O) + assistant header
    Ground truth: Full SOAP note (for generation)
    
    Note: Evaluation will only measure Rouge-L on Assessment + Plan portions.
    """
    client_name = record.get('client_name', 'Unknown')
    dob = record.get('date_of_birth', 'Unknown')
    date = record.get('date', 'Unknown')
    subjective = record.get('subjective', '')
    objective = record.get('objective', '')
    assessment = record.get('assessment', '')
    plan = record.get('plan', '')

    # Prefix: matches training format exactly
    prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a medical documentation assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
Create a SOAP note for:
- Patient: {client_name}
- DOB: {dob}
- Date: {date}
Presentation: {subjective}
Exam: {objective}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    # Ground truth: Full SOAP note (model generates this, but we evaluate only A+P)
    ground_truth = f"""SOAP Note for {client_name}
Subjective: {subjective}
Objective: {objective}
Assessment: {assessment}
Plan: {plan}"""
    
    return prefix, ground_truth


def format_preformatted_prefix(
    record: Dict, 
    tokenizer: AutoTokenizer, 
    split_ratio: float = 0.5
) -> Tuple[str, str]:
    """
    For preformatted text, split by tokens. First half is prefix, second half is target.
    """
    text = record.get('text', '')
    tokens = tokenizer.encode(text, add_special_tokens=False)
    split_idx = int(len(tokens) * split_ratio)
    split_idx = max(1, min(split_idx, len(tokens) - 1))
    
    prefix_text = tokenizer.decode(tokens[:split_idx], skip_special_tokens=True)
    suffix_text = tokenizer.decode(tokens[split_idx:], skip_special_tokens=True)
    
    formatted_prefix = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Continue this text:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{prefix_text}"""
    
    return formatted_prefix, suffix_text


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str, device: str, dtype_str: str):
    dtype = getattr(torch, dtype_str)
    logger.info(f"Loading model from {model_path}")
    
    # Try loading tokenizer from model path, fall back to base Llama if it fails
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
        logger.info("Falling back to base Llama tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", 
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_dataset(path: str) -> List[Dict]:
    logger.info(f"Loading dataset from {path}")
    with open(path, 'r') as f:
        content = f.read().strip()
    
    # Detect format by content: JSON array starts with '[', JSONL doesn't
    if content.startswith('['):
        data = json.loads(content)
    else:
        # JSONL format (one JSON object per line)
        data = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    logger.info(f"Loaded {len(data)} examples")
    return data


# ============================================================================
# Contrasting Generation
# ============================================================================

def contrasting_generate(
    oracle_model: AutoModelForCausalLM,
    unlearned_model: Optional[AutoModelForCausalLM],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: ExtractionConfig,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    """
    Generate using: combined = (1-|γ|)*log_p_unlearned + |γ|*log_p_oracle
    
    γ=1.0: pure oracle | γ=0.0: pure unlearned | γ>1.0: amplified oracle
    
    Token filtering (when minus_value is set):
    Keep tokens where: oracle_log_prob > max_oracle_log_prob - minus_value
    Paper's γ=1e-5 corresponds to minus_value ≈ 11.5
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    gamma = config.gamma
    
    generated_ids = input_ids.clone()
    oracle_past = None
    unlearned_past = None
    
    for _ in range(config.max_new_tokens):
        with torch.no_grad():
            if oracle_past is None:
                oracle_out = oracle_model(input_ids=generated_ids, attention_mask=attention_mask, 
                                          use_cache=True, return_dict=True)
            else:
                oracle_out = oracle_model(input_ids=generated_ids[:, -1:], attention_mask=attention_mask,
                                          past_key_values=oracle_past, use_cache=True, return_dict=True)
            oracle_logits = oracle_out.logits[:, -1, :].float()
            oracle_past = oracle_out.past_key_values
        
        if unlearned_model is not None and gamma != 1.0:
            with torch.no_grad():
                if unlearned_past is None:
                    unlearned_out = unlearned_model(input_ids=generated_ids, attention_mask=attention_mask,
                                                    use_cache=True, return_dict=True)
                else:
                    unlearned_out = unlearned_model(input_ids=generated_ids[:, -1:], attention_mask=attention_mask,
                                                    past_key_values=unlearned_past, use_cache=True, return_dict=True)
                unlearned_logits = unlearned_out.logits[:, -1, :].float()
                unlearned_past = unlearned_out.past_key_values
        else:
            unlearned_logits = oracle_logits
        
        # Log-space interpolation
        abs_gamma = abs(gamma)
        oracle_log_probs = F.log_softmax(oracle_logits, dim=-1)
        unlearned_log_probs = F.log_softmax(unlearned_logits, dim=-1)
        combined_log_probs = (1 - abs_gamma) * unlearned_log_probs + abs_gamma * oracle_log_probs
        probs = torch.exp(combined_log_probs)
        
        if config.do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            if config.minus_value is None:
                # Standard argmax on combined distribution
                next_token = probs.argmax(dim=-1, keepdim=True)
            else:
                # Token filtering (from paper's code):
                # mask = logits0 > (max_logits0 - minus_value)
                # where logits0 is log-softmax (oracle log probs)
                max_oracle_log_prob = oracle_log_probs.max(dim=-1, keepdim=True)[0]
                mask = oracle_log_probs > (max_oracle_log_prob - config.minus_value)
                # Apply mask to combined probs, then argmax
                probs_masked = probs.masked_fill(~mask, 0.0)
                next_token = probs_masked.argmax(dim=-1, keepdim=True)
        
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)], dim=-1)
        
        if (next_token == tokenizer.eos_token_id).all():
            break
    
    return generated_ids


# ============================================================================
# Main
# ============================================================================

def run_extraction(config: ExtractionConfig) -> List[ExtractionResult]:
    torch.manual_seed(config.seed)
    
    oracle_model, tokenizer = load_model(config.oracle_model_path, config.device, config.dtype)
    
    if config.unlearned_model_path and config.gamma != 1.0:
        unlearned_model, _ = load_model(config.unlearned_model_path, config.device, config.dtype)
    else:
        unlearned_model = None
    
    dataset = load_dataset(config.dataset_path)
    data_format = detect_data_format(dataset)
    logger.info(f"Detected format: {data_format}")
    
    results = []
    for idx, record in enumerate(tqdm(dataset, desc="Extracting")):
        if data_format == 'medical_soap':
            prefix, ground_truth = format_medical_soap_prefix(record)
        else:
            prefix, ground_truth = format_preformatted_prefix(record, tokenizer, config.split_ratio)
        
        if not ground_truth.strip():
            continue
        
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=config.max_length).to(config.device)
        input_length = inputs.input_ids.shape[1]
        
        generated_ids = contrasting_generate(oracle_model, unlearned_model, inputs.input_ids, 
                                              inputs.attention_mask, config, tokenizer)
        
        generated_text = tokenizer.decode(generated_ids[0, input_length:], skip_special_tokens=True)
        
        results.append(ExtractionResult(
            idx=idx,
            prefix=prefix,
            ground_truth=ground_truth,
            generated=generated_text,
            data_format=data_format,
            prefix_tokens=input_length,
            gt_tokens=len(tokenizer.encode(ground_truth, add_special_tokens=False)),
            gen_tokens=generated_ids.shape[1] - input_length,
        ))
    
    return results


def save_results(results: List[ExtractionResult], config: ExtractionConfig, output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    output = {"config": asdict(config), "num_examples": len(results), "results": [asdict(r) for r in results]}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extraction attack for Llama 3.1 Instruct models")
    parser.add_argument("--oracle-model", type=str, required=True, help="Path to oracle (fine-tuned) model")
    parser.add_argument("--unlearned-model", type=str, default=None, help="Path to unlearned model (for gamma != 1.0)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON/JSONL")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--gamma", type=float, default=1.0, help="Guidance weight (1.0=oracle, >1.0=amplified)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--max-length", type=int, default=512, help="Max input length")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Prefix ratio for preformatted data")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy")
    parser.add_argument("--minus-value", type=float, default=None, 
                        help="Token filter threshold in log space. Paper's γ=1e-5 ≈ minus_value=11.5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = ExtractionConfig(
        oracle_model_path=args.oracle_model,
        unlearned_model_path=args.unlearned_model,
        dataset_path=args.dataset,
        output_path=args.output,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        split_ratio=args.split_ratio,
        do_sample=args.do_sample,
        minus_value=args.minus_value,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    
    logger.info(f"Config: {asdict(config)}")
    results = run_extraction(config)
    save_results(results, config, args.output)
    logger.info("Done!")


if __name__ == "__main__":
    main()