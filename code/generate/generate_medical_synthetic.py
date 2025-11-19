"""
Synthetic Medical Dataset Generator using Tinker API

Example usage (PowerShell):
python generate_medical_synthetic.py `
--n_record 1000 `
--model "meta-llama/Llama-3.1-8B-Instruct" `
--output "synthetic_soap_notes.jsonl" `
--max_tokens 3000 `
--temperature 0.7 `
--samples_per_prompt 10 `
--api_key "your-tinker-api-key"

Medical Record Schema (SOAP notes format)
MEDICAL_SCHEMA = {
    "client_name": "string - patient's full name",
    "date_of_birth": "string - YYYY-MM-DD format",
    "date": "string - YYYY-MM-DD format (date of visit)",
    "subjective": "string - narrative description of symptoms, medications, lifestyle factors",
    "objective": "string - vital signs, physical exam findings, observations",
    "assessment": "string - likely diagnosis and differential diagnoses",
    "plan": "string - treatment plan, referrals, and follow-up"
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict
import argparse
import tinker

# Default configuration matching paper's experimental setup
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_FILE = "dataset/synthetic_soap_notes.jsonl"
DEFAULT_NUM_RECORDS = 1000
DEFAULT_MAX_TOKENS = 3000  # Reduced from 15000 to generate records that fit in 2048 tokens after formatting
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SAMPLES_PER_PROMPT = 10


# Prompt Template (following paper's text-generation approach)
def make_prompt_template() -> str:
    """
    Create the prompt template for LLM to generate synthetic medical records.
    Following the paper's approach of clear instructions + schema + examples.
    """
    
    prompt = """I would like to generate synthetic medical data for machine learning purposes. Specifically, I would use SOAP notes as the data type. Below is a note template you need to follow, which has client name, date of birth, date, as well as subjective, objective, assessment, and plan. The template is just for you to refer, you do not need to generate each line of the template. Instead, only several lines for each of the SOAP is enough, try not to be too tedious for each record. For each record, please generate with a PII (client name, date of birthday), one person per record. client name: [name] date of birth: [birthday date] date: [visiting date] Subjective: [Description of symptoms, onset of symptoms, location of symptoms, duration of symptoms, characteristics of symptoms, alleviating or aggravating factors, timing, and severity] [Current medications and response to treatment] (write this section in narrative form. Write in full sentences and do not include any bullet points) [Any side effects experienced] (write this section in narrative form. Write in full sentences and do not include any bullet points) [Non-pharmacological interventions tried] (write this section in narrative form. Write in full sentences and do not include any bullet points) [Description of any related lifestyle factors] (write this section in narrative form . Write in full sentences and do not include any bullet points) [Patient’s experience and management of symptoms] (write this section in narrative form. Write in full sentences and do not include any bullet points) [Any recent changes in symptoms or condition] (write this section in narrative form. Write in full sentences and do not include any bullet points) [Any pertinent positive or pertinent negatives in review of systems] (write this section in narrative form. Write in full sentences and do not include any bullet points) Objective: Vital Signs:Blood Pressure: [blood pressure reading] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Heart Rate: [heart rate reading] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Respiratory Rate: [respiratory rate reading] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Temperature: [temperature reading] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Oxygen Saturation: [oxygen saturation reading] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) General Appearance: [general appearance description] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) HEENT: [head, eyes, ears, nose, throat findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Neck: [neck findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Cardiovascular: [cardiovascular findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Respiratory: [respiratory findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Abdomen: [abdominal findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Musculoskeletal: [musculoskeletal findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Neurological: [neurological findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Skin: [skin findings] (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank.) Assessment: [Likely diagnosis] [Differential diagnosis (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank)] Diagnostic Tests: (only include if explicitly mentioned other skip section) [Investigations and tests planned (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank)] Plan: [Treatment planned for Issue 1 (only include if explicitly mentioned in the transcript, contextual notes or clinical note, otherwise leave blank)] [Relevant referrals for Issue 1 (only include if explicitly mentioned- [Likely diagnosis for Issue 1 (condition name only)] (Never come up with your own patient details, assessment, diagnosis, interventions, evaluation or plan for continuing care - use only the transcript, contextual notes, or clinical note as a reference for the information included in your note. If any information related to a placeholder has not been explicitly mentioned in the transcript, contextual notes, or clinical note, you must not state the information has not been explicitly mentioned in your output, just leave the relevant placeholder or section blank). Then, here is an example of SOAP fields for you to refer: Subjective: The patient, a 52-year-old male, presents with a new rash on his back and arms, which he has noticed for the past two weeks. He describes the rash as "itchy and red ," and mentions that it seems to be getting worse despite over-the-counter anti-itch creams. The patient denies any fever, joint pain, or recent exposure to new soaps or detergents. Objective: Appearance: The patient appears well-nourished and in no acute distress. Skin: Exam reveals erythematous, scaly plaques on the back and arms. There is evidence of excoriation due to itching. No signs of systemic involvement. Lesions: Lesions are well-defined, with some areas showing mild papules. No signs of pustules or ulcers. Other Systems: Vital signs are within normal limits. No lymphadenopathy noted. Assessment: The presentation is consistent with psoriasis, characterized by itchy, scaly plaques . The absence of systemic symptoms and well-defined lesions supports this diagnosis. Differential diagnoses include eczema or fungal infection, but these are less likely given the clinical presentation. Plan: Initiate topical treatment with high-potency corticosteroids to reduce inflammation and itching. Recommend emollients to improve skin hydration and prevent dryness. Educate the patient on the nature of psoriasis, including triggers and management strategies. . Suggest lifestyle modifications such as stress management and dietary adjustments to potentially improve symptoms. You should add PII in front of the SOAP. Please generate 10 records for me, in json format, with "client name, date of birth, date, as well as subjective, objective, assessment, and plan" as keys.
"""
    return prompt


# JSON Extraction Utility
def normalize_keys(obj):
    """
    Normalize dictionary keys by replacing spaces with underscores.
    Handles nested dictionaries and lists.
    Also flattens nested SOAP field dictionaries into strings.
    """
    if isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            normalized_key = key.replace(' ', '_').lower()
            
            # Special handling for SOAP fields - convert nested dicts to strings
            if normalized_key in ['subjective', 'objective', 'assessment', 'plan']:
                if isinstance(value, dict):
                    # Flatten nested dict into a readable string
                    parts = []
                    for k, v in value.items():
                        if isinstance(v, str) and v.strip():
                            parts.append(f"{k}: {v}")
                    normalized[normalized_key] = "\n".join(parts) if parts else ""
                elif isinstance(value, str):
                    normalized[normalized_key] = value
                else:
                    normalized[normalized_key] = str(value)
            else:
                normalized[normalized_key] = normalize_keys(value)
        return normalized
    elif isinstance(obj, list):
        return [normalize_keys(item) for item in obj]
    else:
        return obj

def extract_json_from_text(text: str) -> Optional[List[Dict]]:
    """
    Extract JSON from LLM output - handles both single objects and arrays.
    Handles common LLM formatting issues (markdown, extra text, etc.)
    Also normalizes keys by converting spaces to underscores.
    Returns a list of dictionaries (even if input is a single object).
    """
    # Remove common markers
    text = text.replace('```json', '').replace('```', '')
    text = text.replace('<|end_of_text|>', '')
    text = text.strip()
    
    # First, try to find a JSON array
    array_start = text.find('[')
    if array_start != -1:
        # Try to extract JSON array
        stack = []
        for i in range(array_start, len(text)):
            if text[i] == '[':
                stack.append('[')
            elif text[i] == ']':
                if stack:
                    stack.pop()
                if not stack:
                    candidate = text[array_start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            return [normalize_keys(item) for item in parsed]
                    except json.JSONDecodeError:
                        continue
    
    # If no array found, try to find JSON object boundaries
    start = text.find('{')
    if start == -1:
        return None
    
    # Use stack to find matching closing brace
    stack = []
    for i in range(start, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            if stack:
                stack.pop()
            if not stack:
                candidate = text[start:i+1]
                try:
                    parsed = json.loads(candidate)
                    # Normalize keys to handle "client name" -> "client_name"
                    return [normalize_keys(parsed)]  # Return as list
                except json.JSONDecodeError as e:
                    # Try continuing to find another closing brace
                    continue
    
    # If stack-based approach fails, try to find last } and work backwards
    if '}' in text:
        end = text.rfind('}')
        start = text.find('{')
        if start != -1 and end > start:
            candidate = text[start:end+1]
            try:
                parsed = json.loads(candidate)
                return [normalize_keys(parsed)]  # Return as list
            except json.JSONDecodeError:
                pass
    
    return None


# Data Validation
def validate_medical_record(record: Dict) -> bool:
    """
    Validate generated SOAP note record matches schema.
    Returns True if valid, False otherwise.
    """
    required_fields = [
        'client_name', 'date_of_birth', 'date',
        'subjective', 'objective', 'assessment', 'plan'
    ]
    
    # Check all required fields present
    for field in required_fields:
        if field not in record:
            print(f"Missing required field: {field}")
            return False
    
    # Validate client_name is non-empty string
    if not isinstance(record['client_name'], str) or len(record['client_name'].strip()) == 0:
        print(f"client_name must be non-empty string")
        return False
    
    # Validate date_of_birth format - must be YYYY-MM-DD
    if not isinstance(record['date_of_birth'], str):
        print(f"date_of_birth must be string")
        return False
    
    try:
        datetime.strptime(record['date_of_birth'], '%Y-%m-%d')
    except ValueError:
        print(f"date_of_birth must be in YYYY-MM-DD format, got: {record['date_of_birth']}")
        return False
    
    # Validate date format - must be YYYY-MM-DD
    if not isinstance(record['date'], str):
        print(f"date must be string")
        return False
    
    try:
        datetime.strptime(record['date'], '%Y-%m-%d')
    except ValueError:
        print(f"date must be in YYYY-MM-DD format, got: {record['date']}")
        return False
    
    # Validate SOAP note fields are non-empty strings with minimum length requirement
    for soap_field in ['subjective', 'objective', 'assessment', 'plan']:
        if not isinstance(record[soap_field], str):
            print(f"{soap_field} must be string")
            return False
        
        # Strict validation: require meaningful content (at least 20 characters)
        if len(record[soap_field].strip()) < 20:
            print(f"{soap_field} is empty or too short (min 20 chars, got {len(record[soap_field].strip())})")
            return False
    
    return True


# Continue Generation Functions
def count_existing_records(filepath: str) -> int:
    """Count how many records already exist in the output file."""
    if not os.path.exists(filepath):
        return 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"Warning: Could not read existing file: {e}")
        return 0


# Main Generation Function
def generate_synthetic_medical_dataset(
    num_records: int = DEFAULT_NUM_RECORDS,
    base_model: str = DEFAULT_BASE_MODEL,
    output_file: str = DEFAULT_OUTPUT_FILE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    samples_per_prompt: int = DEFAULT_SAMPLES_PER_PROMPT,
    api_key: str = None
):
    """
    Main function to generate synthetic medical dataset using Tinker SDK.
    Follows the paper's methodology: prompt-based generation with sampling.
    
    Args:
        num_records: Total number of records to have in the output file
        base_model: The LLM model to use for generation
        output_file: Path to the output JSONL file
        max_tokens: Maximum tokens to generate per request
        temperature: Sampling temperature (0.0-1.0)
        samples_per_prompt: Number of parallel samples per API call
        api_key: Tinker API key (if not provided, uses environment variable)
    """
    
    print("=" * 80)
    print("SYNTHETIC MEDICAL DATASET GENERATOR (SOAP NOTES)")
    print("Based on methodology from arXiv:2505.24379")
    print("=" * 80)
    print()
    
    # 1. Check existing records
    existing_count = count_existing_records(output_file)
    if existing_count > 0:
        print(f"✓ Found {existing_count} existing records in {output_file}")
        if existing_count >= num_records:
            print(f"✓ Already have {existing_count} records (requested: {num_records})")
            print(f"✓ No additional generation needed!")
            return
        else:
            records_to_generate = num_records - existing_count
            print(f"✓ Will generate {records_to_generate} more records to reach {num_records} total")
            NUM_RECORDS = records_to_generate
    else:
        print(f"✓ Starting fresh - will generate {num_records} records")
        NUM_RECORDS = num_records
    print()
    
    # 2. Check for API key
    if api_key:
        os.environ["TINKER_API_KEY"] = api_key
    
    api_key_env = os.environ.get("TINKER_API_KEY")
    if not api_key_env:
        print("ERROR: TINKER_API_KEY environment variable not set!")
        print()
        print("Please set your Tinker API key:")
        print("  PowerShell:  $env:TINKER_API_KEY = 'your-api-key'")
        print("  Or use --api-key argument")
        return
    
    print(f"Tinker API key found")
    print(f"Target: {NUM_RECORDS} synthetic SOAP notes")
    print(f"Base model: {base_model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Samples per prompt: {samples_per_prompt}")
    print(f"Output: {output_file}")
    print()
    
    # 2. Initialize Tinker client
    print("Initializing Tinker service client...")
    try:
        service_client = tinker.ServiceClient()
        print("✓ Service client initialized")
    except Exception as e:
        print(f"Failed to initialize Tinker client: {e}")
        return
    
    # 3. List available models (optional - for verification)
    try:
        capabilities = service_client.get_server_capabilities()
        for model_info in capabilities.supported_models[:5]:  # Show first 5
            print(f"  - {model_info.model_name}")
        print(f"  ... and {len(capabilities.supported_models) - 5} more")
    except Exception as e:
        print(f"  (Could not list models: {e})")
    
    # 4. Create training client (used for sampling even without training)
    print(f"\nCreating training client with base model: {base_model}...")
    try:
        training_client = service_client.create_lora_training_client(base_model=base_model)
        tokenizer = training_client.get_tokenizer()
        print("✓ Training client created")
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"Failed to create training client: {e}")
        return
    
    # 5. Create a simple sampling client using the base model
    print("\nCreating sampling client from base model...")
    try:
        # Use service client to create sampling client directly
        sampling_client = service_client.create_sampling_client(base_model=base_model)
        print("✓ Sampling client ready")
    except Exception as e:
        print(f"Failed to create sampling client: {e}")
        print("Trying alternative approach...")
        try:
            # Alternative: use save_weights but with no-op training
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"medical-synthetic-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print("✓ Sampling client ready (via training client)")
        except Exception as e2:
            print(f"Also failed: {e2}")
            return
    
    # 6. Generate prompt template
    prompt_template = make_prompt_template()
    print(f"\n✓ Prompt template created ({len(prompt_template)} chars)")
    
    # 7. Begin generation loop
    print("\n" + "=" * 80)
    print("GENERATING RECORDS")
    print("=" * 80)
    print()
    
    generated = 0
    failed = 0
    max_attempts = NUM_RECORDS * 5  # Maximum attempts before giving up
    attempts = 0
    
    with open(output_file, "a", encoding="utf-8") as out_fh:
        
        while generated < NUM_RECORDS and attempts < max_attempts:
            attempts += 1
            print(f"[{generated + 1}/{NUM_RECORDS}] Attempt {attempts}: Generating record...")
            
            # Tokenize prompt
            prompt_tokens = tokenizer.encode(prompt_template, add_special_tokens=True)
            model_input = tinker.types.ModelInput.from_ints(tokens=prompt_tokens)
            
            # Set sampling parameters (matching paper's approach)
            sampling_params = tinker.types.SamplingParams(
                max_new_tokens=max_tokens,  # Use max_new_tokens instead of max_tokens
                temperature=temperature,
                stop=["\n\nWould you like", "\n\nLet me know", "\n\nHere's another", "\n\nBelow is another"]  # Stop after one complete JSON
            )
            
            # Sample from model
            try:
                future = sampling_client.sample(
                    prompt=model_input,
                    sampling_params=sampling_params,
                    num_samples=samples_per_prompt
                )
                result = future.result()
            except Exception as e:
                print(f"Sampling failed: {e}")
                failed += 1
                continue
            
            # Process generated samples
            for seq in result.sequences[:samples_per_prompt]:
                if generated >= NUM_RECORDS:
                    break
                
                # Decode the generated tokens
                # Try decoding all tokens first, then the full text should include both prompt and generation
                try:
                    # Method 1: seq.tokens might already be just generated tokens
                    generated_text = tokenizer.decode(seq.tokens)
                    
                    # If the decoded text starts with our prompt, extract just the generated part
                    if generated_text.startswith("Generate synthetic medical"):
                        # Find where the example ends and new generation begins
                        example_end = generated_text.find('Now generate ONE new SOAP note')
                        if example_end != -1:
                            # Find the end of the instruction line
                            instruction_end = generated_text.find(':', example_end) + 1
                            generated_text = generated_text[instruction_end:].strip()
                except Exception as e:
                    print(f"Decoding failed: {e}")
                    failed += 1
                    continue
                
                # Extract JSON from generated text (returns list of records)
                records = extract_json_from_text(generated_text)
                
                if records is None or len(records) == 0:
                    print(f"Failed to parse JSON")
                    failed += 1
                    continue
                
                # Process each record in the list
                for record in records:
                    if generated >= NUM_RECORDS:
                        break
                    
                    # Validate record
                    if not validate_medical_record(record):
                        print(f"Validation failed")
                        failed += 1
                        continue
                    
                    # Write valid record
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()  # Ensure written to disk
                    generated += 1
                    
                    print(f"Record {existing_count + generated} generated (Client: {record['client_name']})")
                
                if generated >= NUM_RECORDS:
                    break
    
    # Check if we hit max attempts
    if attempts >= max_attempts and generated < NUM_RECORDS:
        print()
        print("WARNING: Reached maximum attempts!")
        print(f"Generated {generated}/{NUM_RECORDS} records after {attempts} attempts")
        print(f"Consider using a larger model or adjusting the prompt")
    
    # 8. Summary
    total_records = existing_count + generated
    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Successfully generated: {generated} new records")
    print(f"Total records in file: {total_records}")
    print(f"Failed attempts: {failed}")
    print(f"Output file: {output_file}")
    print()
    print("You can now load this dataset in Python:")
    print(f"import json")
    print(f"with open('{output_file}', 'r') as f:")
    print(f"soap_notes = [json.loads(line) for line in f]")
    print(f"print(f'Loaded {{len(soap_notes)}} SOAP notes')")
    print()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical SOAP notes using Tinker API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""  
Note: If synthetic_soap_notes.jsonl already exists, this will append to reach the requested total number of records.
        """
    )
    parser.add_argument(
        '--n_record',
        type=int,
        dest='num_records',
        default=DEFAULT_NUM_RECORDS,
        help=f'Total number of SOAP notes to generate (default: {DEFAULT_NUM_RECORDS})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f'Base model to use for generation (default: {DEFAULT_BASE_MODEL})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f'Output JSONL file path (default: {DEFAULT_OUTPUT_FILE})'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens to generate per request (default: {DEFAULT_MAX_TOKENS})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Sampling temperature 0.0-1.0 (default: {DEFAULT_TEMPERATURE})'
    )
    parser.add_argument(
        '--samples_per_prompt',
        type=int,
        default=DEFAULT_SAMPLES_PER_PROMPT,
        help=f'Number of parallel samples per API call (default: {DEFAULT_SAMPLES_PER_PROMPT})'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='Tinker API key (default: reads from TINKER_API_KEY environment variable)'
    )
    
    args = parser.parse_args()
    
    # Validate num_records
    if args.num_records <= 0:
        print(f"Error: Number of records must be positive (got {args.num_records})")
        exit(1)
    
    # Validate temperature
    if not (0.0 <= args.temperature <= 2.0):
        print(f"Error: Temperature must be between 0.0 and 2.0 (got {args.temperature})")
        exit(1)
    
    generate_synthetic_medical_dataset(
        num_records=args.num_records,
        base_model=args.model,
        output_file=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        samples_per_prompt=args.samples_per_prompt,
        api_key=args.api_key
    )