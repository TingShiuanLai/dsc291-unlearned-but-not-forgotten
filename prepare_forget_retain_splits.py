"""
Prepare forget/retain splits for machine unlearning experiments
Following the methodology from "Unlearned but Not Forgotten" paper
"""

import json
import random
from pathlib import Path

def load_jsonl(filepath):
    """Load JSONL file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, filepath):
    """Save to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def save_json(data, filepath):
    """Save to JSON file (for compatibility with MUSE framework)"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def create_forget_retain_splits(input_file, forget_ratios=[0.10], seed=42, output_dir='dataset'):
    """
    Create forget/retain splits following MUSE paper methodology
    
    Args:
        input_file: Path to synthetic_soap_notes.jsonl
        forget_ratios: List of ratios (e.g., [0.01, 0.05, 0.10])
        seed: Random seed for reproducibility
        output_dir: Directory to save output files (default: 'dataset')
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # Load full dataset
    print(f"\nLoading dataset from {input_file}...")
    full_data = load_jsonl(input_file)
    total_records = len(full_data)
    print(f"Loaded {total_records} SOAP notes")
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle data
    shuffled_data = full_data.copy()
    random.shuffle(shuffled_data)
    
    # Save full dataset in JSON format (MUSE format)
    full_file = output_path / 'full.json'
    print(f"\nSaving {full_file}...")
    save_json(shuffled_data, full_file)
    
    # Create splits for each forget ratio
    for ratio in forget_ratios:
        forget_size = int(total_records * ratio)
        forget_name = str(int(ratio * 100)).zfill(2)  # e.g., "10" for 10%
        
        print(f"\n--- Creating forget{forget_name} split ({ratio*100}% = {forget_size} records) ---")
        
        # Split data
        forget_data = shuffled_data[:forget_size]
        retain_data = shuffled_data[forget_size:]
        
        # Save forget set
        forget_file = output_path / f'forget{forget_name}.json'
        print(f"Saving {forget_file} ({len(forget_data)} records)...")
        save_json(forget_data, forget_file)
        
        # Save retain set  
        retain_file = output_path / f'full_minus_forget{forget_name}.json'
        print(f"Saving {retain_file} ({len(retain_data)} records)...")
        save_json(retain_data, retain_file)
    
    print("\n✓ Data splitting complete!")
    print(f"\nAll files saved to: {output_path.absolute()}")
    print("\nNext steps:")
    print("1. Fine-tune model on dataset/full.json (create oracle model)")
    print("2. Apply unlearning on dataset/forget10.json")
    print("3. Evaluate forget quality and data extraction")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "synthetic_soap_notes_unique_names.jsonl"
    
    # Multiple forget ratios for comprehensive testing (like the paper)
    # Start with forget10 (10%) as the main benchmark
    # Uncomment others to test scalability across different forget set sizes
    FORGET_RATIOS = [
        0.10,  # Standard benchmark (10%) - START HERE
        0.01,  # Single patient scenario (1%)
        0.05,  # Small-scale deletion (5%)
        0.20,  # Large-scale deletion (20%)
    ]
    
    SEED = 42  # For reproducibility
    
    # Create splits
    create_forget_retain_splits(
        input_file=INPUT_FILE,
        forget_ratios=FORGET_RATIOS,
        seed=SEED
    )
