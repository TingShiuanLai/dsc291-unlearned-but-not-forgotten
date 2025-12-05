"""Merge a PEFT LoRA adapter (safetensors) with a base Hugging Face model.

Usage examples:

1) Extract the tar (if not already extracted):
   tar -xf models/medical_model_weights.tar -C models/medical_adapter

2) Merge adapter into a new model dir (this creates a full model that
   `AutoModelForCausalLM.from_pretrained` can load):
   python code/train/merge_adapter.py \
       --adapter-dir models/llama-3.1-8b-medical \
       --base-model meta-llama/Llama-3.1-8B-Instruct \
       --output-dir models/llama-3.1-8b-medical-merged

Notes:
- Requires `transformers`, `peft`, and `safetensors` (see repository `requirements.txt`).
- You will likely need a GPU and an HF token for gated models (pass via HF_TOKEN env var
  or rely on local cached base model).
"""

import argparse
import os
import json
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a local .env file (if present)
load_dotenv()


def merge_adapter(
    adapter_dir: str,
    base_model: str,
    output_dir: str,
    use_8bit: bool = False,
    hf_token: str = None
):
    """Load base model, apply adapter from `adapter_dir`, merge and save to `output_dir`.

    adapter_dir should contain `adapter_model.safetensors` and `adapter_config.json` (PEFT format).
    The LoRA config is automatically loaded from adapter_config.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prefer token provided as argument, fall back to .env / env vars.
    hf_token = (
        hf_token
        or os.environ.get('HF_API_TOKEN')
        or os.environ.get('HF_TOKEN_API')
        or os.environ.get('HF_TOKEN')
    )

    logger.info(f"Loading tokenizer from base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading base model: {base_model}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16
    }
    if use_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        **model_kwargs
    )

    logger.info(f"Applying PEFT adapter from: {adapter_dir}")

    # Load and filter adapter config to remove unsupported fields
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)

    # Get valid LoraConfig parameters
    import inspect
    valid_params = set(inspect.signature(LoraConfig.__init__).parameters.keys())
    valid_params.discard('self')

    # Filter out unsupported fields
    filtered_config = {k: v for k, v in adapter_config.items() if k in valid_params}
    removed_fields = set(adapter_config.keys()) - set(filtered_config.keys())
    if removed_fields:
        logger.warning(f"Removed unsupported config fields: {removed_fields}")

    # Create LoRA config with only supported parameters
    lora_config = LoraConfig(**filtered_config)

    # Load adapter with filtered config
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        config=lora_config,
        device_map="auto"
    )

    logger.info("Merging adapter into base model (this writes merged weights into memory)")
    try:
        merged = peft_model.merge_and_unload()
    except Exception as e:
        logger.error("merge_and_unload failed: %s", e)
        logger.info("Falling back to saving peft model wrapper (adapter-only) to output directory")
        peft_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Adapter-only saved to: {output_dir}")
        return

    logger.info(f"Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merge + save complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge PEFT LoRA adapter into a full model directory"
    )
    parser.add_argument(
        '--adapter-dir',
        type=str,
        required=True,
        help='Path to extracted adapter folder (contains adapter_model.safetensors)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        required=True,
        help='Base model HF id or local path (e.g. meta-llama/Llama-3.1-8B-Instruct)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save merged model to'
    )
    parser.add_argument(
        '--use-8bit',
        action='store_true',
        help='Load base model in 8-bit if desired'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='Hugging Face token (overrides HF_TOKEN env var)'
    )

    args = parser.parse_args()
    merge_adapter(
        args.adapter_dir,
        args.base_model,
        args.output_dir,
        use_8bit=args.use_8bit,
        hf_token=args.hf_token
    )