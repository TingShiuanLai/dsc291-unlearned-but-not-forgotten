"""
Fine-tune Llama-3.1-8B-Instruct on Full Dataset using Hugging Face Transformers and LoRA

This script implements the pre-unlearning fine-tuning step from the paper
"Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLMs"
using Hugging Face Transformers for training and PEFT for LoRA fine-tuning.

Key Features:
- Uses Hugging Face Transformers Trainer for efficient training
- LoRA (Low-Rank Adaptation) via PEFT for parameter-efficient fine-tuning
- No API key required - runs locally on your GPU
- Supports 8-bit quantization for reduced memory usage
- Compatible with all Hugging Face models
- Supports medical dataset formatting

Usage (PowerShell):
    python finetune_hf_lora.py `
        --data_file "dataset/full.json" `
        --base_model "meta-llama/Llama-3.1-8B-Instruct" `
        --output_dir "models/llama-3.1-8b-medical-oracle" `
        --num_epochs 3 `
        --batch_size 8 `
        --learning_rate 2e-5 `
        --lora_rank 16 `
        --lora_alpha 32

Requirements:
    pip install torch transformers peft datasets accelerate bitsandbytes
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default hyperparameters based on paper and standard practice
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATA_FILE = "dataset/full.json"
DEFAULT_OUTPUT_DIR = "models/llama-3.1-8b-medical-oracle"
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_WARMUP_STEPS = 100
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1


def load_dataset_from_file(file_path: str) -> List[Dict]:
    """
    Load dataset from JSON file (either .json or .jsonl format).
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        List of dictionaries containing medical records
    """
    logger.info(f"Loading dataset from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    records = []
    
    # Try to detect format automatically
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)
        
        # Check if it looks like JSONL
        is_jsonl = False
        if first_line:
            try:
                json.loads(first_line)
                second_line = f.readline().strip()
                if second_line:
                    try:
                        json.loads(second_line)
                        is_jsonl = True
                    except:
                        pass
                f.seek(0)
            except:
                f.seek(0)
    
    # Load based on detected format
    if is_jsonl or file_path.endswith('.jsonl'):
        logger.info("Detected format: JSONL (JSON Lines)")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
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


def format_record_as_conversation(record: Dict) -> str:
    """
    Format a medical record as a text sequence for fine-tuning.
    
    Following instruction-tuning format compatible with Llama-3.1-Instruct:
    - System instruction
    - Patient presentation (as user input)
    - SOAP note (as assistant output)
    
    Args:
        record: Dictionary containing SOAP note fields
        
    Returns:
        Formatted text string
    """
    # Extract fields with defaults
    client_name = record.get('client_name', 'Unknown')
    dob = record.get('date_of_birth', 'Unknown')
    date = record.get('date', 'Unknown')
    subjective = record.get('subjective', '')
    objective = record.get('objective', '')
    assessment = record.get('assessment', '')
    plan = record.get('plan', '')
    
    # Format as instruction-following conversation
    formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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
    
    return formatted


def prepare_dataset(records: List[Dict], tokenizer, max_length: int = 512) -> Dataset:
    """
    Prepare dataset for training with tokenization.
    
    Args:
        records: List of medical records
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Hugging Face Dataset object
    """
    logger.info(f"Preparing {len(records)} records for training...")
    
    # Format all records
    formatted_texts = []
    for idx, record in enumerate(records, 1):
        try:
            formatted_text = format_record_as_conversation(record)
            formatted_texts.append(formatted_text)
            
            if idx % 100 == 0:
                logger.info(f"Formatted {idx}/{len(records)} records...")
        except Exception as e:
            logger.warning(f"Failed to format record {idx}: {e}")
    
    logger.info(f"Successfully formatted {len(formatted_texts)} records")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_texts})
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
    return tokenized_dataset


def setup_model_and_tokenizer(
    base_model: str,
    lora_rank: int,
    lora_alpha: int,
    use_8bit: bool = False
) -> tuple:
    """
    Load base model and tokenizer, and apply LoRA configuration.
    
    Args:
        base_model: Model identifier from Hugging Face
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        use_8bit: Whether to use 8-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer from: {base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading model from: {base_model}")
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    if use_8bit:
        logger.info("Using 8-bit quantization")
        model_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        **model_kwargs
    )
    
    # Prepare model for k-bit training if using quantization
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info(f"Configuring LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def finetune_model(
    data_file: str,
    base_model: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    max_seq_length: int,
    warmup_steps: int,
    gradient_accumulation_steps: int,
    use_8bit: bool = False,
    save_steps: int = 500,
    logging_steps: int = 10
):
    """
    Fine-tune Llama-3.1-8B-Instruct on medical dataset using LoRA.
    
    Args:
        data_file: Path to training data JSON file
        base_model: Base model identifier
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        learning_rate: Learning rate for optimizer
        lora_rank: LoRA rank (r parameter)
        lora_alpha: LoRA alpha (scaling parameter)
        max_seq_length: Maximum sequence length
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        use_8bit: Whether to use 8-bit quantization
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
    """
    print("\n" + "=" * 80)
    print("FINE-TUNING LLAMA-3.1-8B-INSTRUCT WITH HUGGING FACE TRANSFORMERS AND LORA")
    print("=" * 80)
    print()
    print(f"Base Model: {base_model}")
    print(f"Training Data: {data_file}")
    print(f"Output Directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"LoRA Alpha: {lora_alpha}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"8-bit Quantization: {use_8bit}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    records = load_dataset_from_file(data_file)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        base_model=base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_8bit=use_8bit
    )
    
    # Prepare dataset
    train_dataset = prepare_dataset(records, tokenizer, max_seq_length)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        report_to="none",
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Print training info
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Total examples: {len(train_dataset)}")
    logger.info(f"Batch size per device: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps * torch.cuda.device_count()}")
    logger.info(f"Steps per epoch: {len(train_dataset) // (batch_size * gradient_accumulation_steps)}")
    logger.info(f"Total training steps: {(len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs}")
    logger.info("=" * 80)
    
    # Start training
    logger.info("\nStarting training...")
    trainer.train()
    
    # Save final model
    logger.info("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save LoRA config
    lora_config_path = os.path.join(output_dir, "lora_config.json")
    with open(lora_config_path, 'w') as f:
        json.dump({
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "base_model": base_model,
            "training_completed": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\nModel saved to: {output_dir}")
    logger.info("\nTraining complete!")
    
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Use checkpoint in '{output_dir}' as oracle model")
    print("2. Fine-tune on forget sets (forget01, forget05, forget10, forget20)")
    print("3. Apply unlearning methods")
    print("4. Evaluate data extraction attacks")
    print()
    print("Model is ready for unlearning experiments!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B-Instruct using Hugging Face Transformers and LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  PowerShell:
    python finetune_hf_lora.py `
        --data_file "dataset/full.json" `
        --base_model "meta-llama/Llama-3.1-8B-Instruct" `
        --output_dir "models/llama-3.1-8b-medical-oracle" `
        --num_epochs 3 `
        --batch_size 4 `
        --learning_rate 2e-5

  With 8-bit quantization for lower memory usage:
    python finetune_hf_lora.py `
        --data_file "dataset/full.json" `
        --use_8bit
        """
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        default=DEFAULT_DATA_FILE,
        help=f'Path to training data file (JSON or JSONL) (default: {DEFAULT_DATA_FILE})'
    )
    
    parser.add_argument(
        '--base_model',
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f'Base model to fine-tune (default: {DEFAULT_BASE_MODEL})'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save model checkpoints (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f'Number of training epochs (default: {DEFAULT_NUM_EPOCHS})'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Training batch size per device (default: {DEFAULT_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=DEFAULT_LORA_RANK,
        help=f'LoRA rank parameter (default: {DEFAULT_LORA_RANK})'
    )
    
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help=f'LoRA alpha parameter (default: {DEFAULT_LORA_ALPHA})'
    )
    
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f'Maximum sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})'
    )
    
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f'Number of warmup steps (default: {DEFAULT_WARMUP_STEPS})'
    )
    
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help=f'Gradient accumulation steps (default: {DEFAULT_GRADIENT_ACCUMULATION_STEPS})'
    )
    
    parser.add_argument(
        '--use_8bit',
        action='store_true',
        help='Use 8-bit quantization for lower memory usage'
    )
    
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='Save checkpoint every N steps (default: 500)'
    )
    
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=10,
        help='Log metrics every N steps (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Run fine-tuning
    finetune_model(
        data_file=args.data_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_8bit=args.use_8bit,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps
    )


if __name__ == '__main__':
    main()
