"""
Fine-tune models using Tinker API with LoRA - Standalone Version

This script uses only the core tinker API without tinker_cookbook dependencies.
All helper functions are implemented directly in this file.

Usage:
    python finetune_tinker_lora.py \
        --data_file "dataset/wmdp_bio_full.json" \
        --data_format "preformatted" \
        --base_model "meta-llama/Llama-3.1-8B-Instruct" \
        --output_dir "checkpoints/llama-3.1-8b-wmdp" \
        --num_epochs 3 \
        --batch_size 8 \
        --learning_rate 2e-5 \
        --lora_rank 8
"""

import os
import json
import argparse
import logging
import time
import math
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default hyperparameters
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATA_FILE = "../../dataset/wmdp_bio_full.json"
DEFAULT_OUTPUT_DIR = "../../checkpoints/llama-3.1-8b-wmdp-full"
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_WARMUP_STEPS = 100
DEFAULT_SAVE_EVERY = 100


# ============================================================================
# Helper Functions (replacing tinker_cookbook dependencies)
# ============================================================================

def compute_mean_nll(logprobs_list: List, weights_list: List) -> float:
    """Compute mean negative log-likelihood."""
    all_logprobs = []
    all_weights = []
    
    for logprobs, weights in zip(logprobs_list, weights_list):
        if hasattr(logprobs, 'tolist'):
            logprobs = logprobs.tolist()
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        all_logprobs.extend(logprobs)
        all_weights.extend(weights)
    
    all_logprobs = np.array(all_logprobs)
    all_weights = np.array(all_weights)
    
    if all_weights.sum() > 0:
        return -np.dot(all_logprobs, all_weights) / all_weights.sum()
    else:
        return 0.0


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset_from_file(file_path: str) -> List[Dict]:
    """Load dataset from JSON file (either .json or .jsonl format)."""
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
    
    # Load based on format
    if is_jsonl or file_path.endswith('.jsonl'):
        logger.info("Detected format: JSONL")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
    else:
        logger.info("Detected format: JSON array")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                raise ValueError("JSON file must contain an array of records")
    
    logger.info(f"Loaded {len(records)} records")
    return records


def detect_data_format(records: List[Dict]) -> str:
    """Auto-detect the data format."""
    if not records:
        raise ValueError("Empty dataset")

    keys = set(records[0].keys())

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
        logger.warning(f"Unknown format with keys: {keys}")
        return 'generic'


# ============================================================================
# Data Formatting (Llama 3 Chat Template)
# ============================================================================

def format_medical_soap(record: Dict) -> str:
    client_name = record.get('client_name', 'Unknown')
    dob = record.get('date_of_birth', 'Unknown')
    date = record.get('date', 'Unknown')
    subjective = record.get('subjective', '')
    objective = record.get('objective', '')
    assessment = record.get('assessment', '')
    plan = record.get('plan', '')

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical documentation assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Create a SOAP note for:
- Patient: {client_name}
- DOB: {dob}
- Date: {date}

Presentation: {subjective}

Exam: {objective}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

SOAP Note for {client_name}

Subjective: {subjective}

Objective: {objective}

Assessment: {assessment}

Plan: {plan}<|eot_id|>"""


def format_instruction(record: Dict) -> str:
    instruction = record.get('instruction', '')
    output = record.get('output') or record.get('response', '')
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""


def format_qa(record: Dict) -> str:
    question = record.get('question', '')
    answer = record.get('answer', '')
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""


def format_prompt_completion(record: Dict) -> str:
    prompt = record.get('prompt', '')
    completion = record.get('completion', '')
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{completion}<|eot_id|>"""


def format_preformatted(record: Dict) -> str:
    text = record.get('text', '')
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Continue this text:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{text}<|eot_id|>"""


def format_chat(record: Dict) -> str:
    messages = record.get('messages', [])
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    return formatted


def format_generic(record: Dict) -> str:
    for field in ['text', 'content', 'data', 'example', 'input']:
        if field in record:
            text = record[field]
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Continue this text:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{text}<|eot_id|>"""
    
    for key, value in record.items():
        if isinstance(value, str) and value.strip():
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Continue this text:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{value}<|eot_id|>"""
    
    raise ValueError(f"Could not find text field in record: {record}")


def format_record(record: Dict, data_format: str) -> str:
    formatters = {
        'medical_soap': format_medical_soap,
        'instruction': format_instruction,
        'qa': format_qa,
        'prompt_completion': format_prompt_completion,
        'preformatted': format_preformatted,
        'chat': format_chat,
        'generic': format_generic,
    }
    
    formatter = formatters.get(data_format)
    if not formatter:
        raise ValueError(f"Unknown data format: {data_format}")
    
    return formatter(record)


# ============================================================================
# Datum Conversion
# ============================================================================

def text_to_datum(text: str, tokenizer, max_seq_length: int, 
                  train_on_all: bool = True) -> types.Datum:
    """Convert formatted text to Tinker Datum."""
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Truncate
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
    
    # Create input/target pairs
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    
    # Create weights
    if train_on_all:
        weights = [1.0] * len(target_tokens)
    else:
        # Train only on assistant responses
        weights = [1.0] * len(target_tokens)
        text_lower = text.lower()
        assistant_start = text_lower.find("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:
            tokens_before = len(tokenizer.encode(text[:assistant_start], add_special_tokens=False))
            for i in range(min(tokens_before, len(weights))):
                weights[i] = 0.0
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


# ============================================================================
# Main Training Function
# ============================================================================
@dataclass
class SubmittedBatch:
    """Container for a submitted training batch."""
    fwd_bwd_future: tinker.APIFuture
    optim_step_future: tinker.APIFuture
    batch_datums: List[types.Datum]
    step: int
    epoch: int
    batch_idx: int
    current_lr: float
    start_time: float



async def finetune_model(
    data_file: str,
    base_model: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    max_seq_length: int,
    warmup_steps: int,
    data_format: Optional[str],
    save_every: int,
    base_url: Optional[str],
    resume: bool,
    train_on_all: bool,
):
    """
    Async fine-tuning with pipelining for optimal clock cycle utilization.
    
    Pipelining pattern:
    1. Submit batch N (forward_backward + optim_step)
    2. Wait for batch N-1 to complete (if exists)
    3. Process results and save checkpoints
    4. Loop to submit batch N+1
    
    This ensures we always have a request queued for the next clock cycle.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    records = load_dataset_from_file(data_file)
    
    if data_format is None:
        data_format = detect_data_format(records)
        logger.info(f"Auto-detected data format: {data_format}")
    else:
        logger.info(f"Using specified data format: {data_format}")

    # Create Tinker service client
    service_client = tinker.ServiceClient(base_url=base_url)
    
    # Handle resuming
    start_epoch = 0
    start_batch_in_epoch = 0
    
    if resume:
        resume_info = checkpoint_utils.get_last_checkpoint(output_dir)
        if resume_info:
            logger.info(f"Resuming from: {resume_info['state_path']}")
            training_client = await service_client.create_training_client_from_state_async(
                resume_info["state_path"]
            )
            start_epoch = resume_info.get("epoch", 0)
            start_batch_in_epoch = resume_info.get("batch_in_epoch", 0)
            logger.info(f"Resuming from epoch {start_epoch}, batch {start_batch_in_epoch}")
        else:
            logger.info("No checkpoint found, starting fresh")
            training_client = await service_client.create_lora_training_client_async(
                base_model=base_model, rank=lora_rank
            )
    else:
        logger.info("Creating new training client")
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=lora_rank
        )
    
    # Get tokenizer from training client
    logger.info("Loading tokenizer from training client")
    tokenizer = training_client.get_tokenizer()
    
    # Calculate training steps
    n_batches_per_epoch = len(records) // batch_size
    total_steps = n_batches_per_epoch * num_epochs
    
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Base model: {base_model}")
    logger.info(f"Data format: {data_format}")
    logger.info(f"Total examples: {len(records)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batches per epoch: {n_batches_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"LoRA rank: {lora_rank}")
    logger.info(f"Max seq length: {max_seq_length}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Save every: {save_every} steps")
    logger.info(f"Train on all tokens: {train_on_all}")
    logger.info("=" * 80)
    
    # Save configuration
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'base_model': base_model,
            'data_file': data_file,
            'data_format': data_format,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'max_seq_length': max_seq_length,
            'warmup_steps': warmup_steps,
            'save_every': save_every,
            'train_on_all': train_on_all,
            'training_started': datetime.now().isoformat(),
        }, f, indent=2)
    
    # Training loop with pipelining
    global_step = start_epoch * n_batches_per_epoch + start_batch_in_epoch
    pending_batch: Optional[SubmittedBatch] = None
    
    import random
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        
        # Shuffle data at the start of each epoch
        if epoch > start_epoch or start_batch_in_epoch == 0:
            random.seed(epoch)
            random.shuffle(records)
        
        start_batch = start_batch_in_epoch if epoch == start_epoch else 0
        
        for batch_idx in range(start_batch, n_batches_per_epoch):
            batch_start_time = time.time()
            
            # Learning rate schedule (warmup + cosine)
            if global_step < warmup_steps:
                lr_mult = global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr_mult = max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))
            
            current_lr = learning_rate * lr_mult
            
            # Get batch
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(records))
            batch_records = records[batch_start:batch_end]
            
            # Convert to datums
            batch_datums = []
            for record in batch_records:
                try:
                    formatted_text = format_record(record, data_format)
                    datum = text_to_datum(formatted_text, tokenizer, max_seq_length, train_on_all)
                    batch_datums.append(datum)
                except Exception as e:
                    logger.warning(f"Failed to process record: {e}")
                    continue
            
            if not batch_datums:
                logger.warning(f"Empty batch at step {global_step}, skipping")
                continue
            
            # ===================================================================
            # PIPELINING: Submit next batch BEFORE finishing previous batch
            # ===================================================================
            
            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8
            )
            
            # Submit the current batch (async, non-blocking)
            fwd_bwd_future = await training_client.forward_backward_async(
                batch_datums, 
                loss_fn="cross_entropy"
            )
            optim_step_future = await training_client.optim_step_async(adam_params)
            
            # Store current batch info
            current_batch = SubmittedBatch(
                fwd_bwd_future=fwd_bwd_future,
                optim_step_future=optim_step_future,
                batch_datums=batch_datums,
                step=global_step,
                epoch=epoch,
                batch_idx=batch_idx,
                current_lr=current_lr,
                start_time=batch_start_time,
            )
            
            # Now finish the PREVIOUS batch (if it exists)
            if pending_batch is not None:
                await finish_batch(
                    pending_batch, 
                    training_client, 
                    output_dir, 
                    save_every, 
                    total_steps,
                    n_batches_per_epoch
                )
            
            # Current batch becomes pending for next iteration
            pending_batch = current_batch
            global_step += 1
        
        # Reset for next epoch
        if epoch == start_epoch:
            start_batch_in_epoch = 0
    
    # Finish the last pending batch
    if pending_batch is not None:
        await finish_batch(
            pending_batch, 
            training_client, 
            output_dir, 
            save_every, 
            total_steps,
            n_batches_per_epoch
        )
    
    # Save final checkpoint with both state and weights
    logger.info("\nSaving final checkpoint...")
    print("\n💾 Saving final checkpoint...")
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=output_dir,
        kind="both",  # Save both state (for resuming) and sampler (for inference)
        loop_state={
            "epoch": num_epochs,
            "batch_in_epoch": 0,
            "global_step": global_step,
            "training_completed": datetime.now().isoformat(),
        },
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Checkpoints: {output_dir}")
    logger.info(f"Checkpoint tracking: {os.path.join(output_dir, 'checkpoints.jsonl')}")
    logger.info(f"Total steps: {global_step}")
    logger.info("=" * 80)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Checkpoints: {output_dir}")
    print(f"Total steps: {global_step}")
    print("\nCheckpoint files saved:")
    print(f"  - State checkpoints: step_NNNNNN/ (for resuming)")
    print(f"  - Final sampler: final/ (for inference)")
    print(f"  - Tracking file: checkpoints.jsonl")
    print("=" * 80)


async def finish_batch(
    batch: SubmittedBatch,
    training_client: tinker.TrainingClient,
    output_dir: str,
    save_every: int,
    total_steps: int,
    n_batches_per_epoch: int,
):
    """
    Finish a submitted batch by waiting for results and processing metrics.
    This is called for the PREVIOUS batch while the NEXT batch is being submitted.
    """
    # Wait for forward-backward and optimizer step to complete
    fwd_bwd_result = await batch.fwd_bwd_future.result_async()
    await batch.optim_step_future.result_async()
    
    # Compute metrics
    train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
    train_weights = [d.loss_fn_inputs["weights"] for d in batch.batch_datums]
    train_nll = compute_mean_nll(train_logprobs, train_weights)
    
    batch_time = time.time() - batch.start_time
    
    # Log progress
    if batch.step % 10 == 0 or batch.step == total_steps - 1:
        logger.info(
            f"Step {batch.step}/{total_steps} | "
            f"Epoch {batch.epoch + 1} | "
            f"Batch {batch.batch_idx + 1}/{n_batches_per_epoch} | "
            f"Loss: {train_nll:.4f} | "
            f"LR: {batch.current_lr:.2e} | "
            f"Time: {batch_time:.2f}s"
        )
    
    # Save checkpoint
    if batch.step % save_every == 0 and batch.step > 0:
        print(f"\n💾 Saving checkpoint at step {batch.step}...")
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"step_{batch.step:06d}",
            log_path=output_dir,
            kind="state",  # Save state for resuming
            loop_state={
                "epoch": batch.epoch,
                "batch_in_epoch": batch.batch_idx,
                "global_step": batch.step,
                "loss": float(train_nll),
                "learning_rate": float(batch.current_lr),
            },
        )
        print(f"✓ Checkpoint saved!\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with Tinker API (Async + Pipelined)")
    
    parser.add_argument('--data_file', type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument('--base_model', type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--lora_rank', type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument('--max_seq_length', type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument('--data_format', type=str, default=None,
                       choices=['medical_soap', 'instruction', 'qa', 'prompt_completion', 
                               'preformatted', 'chat', 'generic'])
    parser.add_argument('--save_every', type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train_on_all', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Run async function
    asyncio.run(finetune_model(**vars(args)))


if __name__ == '__main__':
    main()