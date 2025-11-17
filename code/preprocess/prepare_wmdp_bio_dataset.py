"""
Download and process WMDP-Bio-Retain corpus from Hugging Face
Creates forget/retain splits for machine unlearning experiments
Following the methodology from "Unlearned but Not Forgotten" paper
"""

import json
import random
import argparse
import re
import unicodedata
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset

# Default configuration
DEFAULT_TARGET_SIZE = 5300  # ~5.3k sentences
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = 'dataset'
DEFAULT_FORGET_RATIOS = [0.01, 0.05, 0.10, 0.20]

def download_wmdp_bio_retain():
    """
    Download WMDP-Bio-Retain corpus (PubMed papers) from Hugging Face.
    Returns the dataset object.
    """
    print("Downloading WMDP-Bio-Retain corpus from Hugging Face...")
    try:
        # Load the bio-retain-corpus which contains PubMed papers
        dataset = load_dataset("cais/wmdp-corpora", "bio-retain-corpus")
        print(f"✓ Dataset downloaded successfully")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure you have the 'datasets' library installed:")
        print("  pip install datasets")
        return None

def clean_text(text: str) -> str:
    """
    Clean text to remove artifacts that harm fine-tuning quality.

    Removes:
    - Invisible unicode characters (zero-width spaces, etc.)
    - Control characters
    - Weird line terminators
    - Extra whitespace and newlines
    - Table-like structures
    - Common PDF extraction artifacts

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text suitable for fine-tuning
    """
    if not text:
        return ""

    # Normalize unicode to NFC form (canonical composition)
    text = unicodedata.normalize('NFC', text)

    # Remove invisible unicode characters
    # Zero-width spaces, zero-width joiners, zero-width non-joiners, etc.
    invisible_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\ufeff',  # Zero-width no-break space / BOM
        '\u202a',  # Left-to-right embedding
        '\u202b',  # Right-to-left embedding
        '\u202c',  # Pop directional formatting
        '\u202d',  # Left-to-right override
        '\u202e',  # Right-to-left override
    ]
    for char in invisible_chars:
        text = text.replace(char, '')

    # Remove control characters except newline and tab (which we'll handle separately)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')

    # Replace various line terminators with space
    text = text.replace('\r\n', ' ')  # Windows line ending
    text = text.replace('\r', ' ')     # Old Mac line ending
    text = text.replace('\n', ' ')     # Unix line ending
    text = text.replace('\v', ' ')     # Vertical tab
    text = text.replace('\f', ' ')     # Form feed
    text = text.replace('\x85', ' ')   # Next line (NEL)
    text = text.replace('\u2028', ' ') # Line separator
    text = text.replace('\u2029', ' ') # Paragraph separator

    # Remove table-like structures (lines with multiple tabs or pipes)
    # Split by potential table row indicators
    lines = text.split('  ')
    cleaned_lines = []
    for line in lines:
        # Skip lines that look like table rows (multiple tabs or pipes)
        if line.count('\t') >= 2 or line.count('|') >= 3:
            continue
        # Skip lines with repeated dashes or equals (table borders)
        if re.search(r'[-=_]{4,}', line):
            continue
        cleaned_lines.append(line)
    text = ' '.join(cleaned_lines)

    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Remove common PDF artifacts
    text = re.sub(r'\[?\d+\]', '', text)  # Remove citation numbers like [1] or 1
    text = re.sub(r'et al\.?\s*,?', 'et al.', text)  # Normalize "et al"

    # Remove multiple consecutive spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove soft hyphens used for word breaks
    text = text.replace('\u00ad', '')

    # Fix common spacing issues around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation if missing

    return text

def is_valid_sentence(text: str, min_length: int = 30, max_length: int = 1000) -> bool:
    """
    Check if a sentence is valid for fine-tuning.

    Args:
        text: Sentence text
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        True if sentence is valid, False otherwise
    """
    if not text:
        return False

    # Length checks
    if len(text) < min_length or len(text) > max_length:
        return False

    # Must contain at least some alphabetic characters
    if not re.search(r'[a-zA-Z]', text):
        return False

    # Reject if too many special characters (likely corrupted)
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,;:!?()\-\']', text)) / len(text)
    if special_char_ratio > 0.15:  # More than 15% special chars
        return False

    # Reject if too many digits (likely tables or figure captions)
    digit_ratio = len(re.findall(r'\d', text)) / len(text)
    if digit_ratio > 0.3:  # More than 30% digits
        return False

    # Must end with proper punctuation
    if not re.search(r'[.!?]$', text.strip()):
        return False

    return True

def extract_sentences_from_dataset(dataset, target_size: int, seed: int = 42) -> List[Dict]:
    """
    Extract and sample sentences from the WMDP bio-retain corpus (PubMed papers).

    Args:
        dataset: HuggingFace dataset object containing PubMed papers
        target_size: Target number of sentences (~5.3k)
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries with sentence data
    """
    print(f"\nExtracting sentences from PubMed papers...")

    random.seed(seed)
    sentences = []
    filtered_count = 0

    # The bio-retain-corpus only has a 'train' split
    split_data = dataset['train']
    print(f"Processing train split ({len(split_data)} papers)...")

    for idx, paper in enumerate(split_data):
        # The bio-retain-corpus only contains 'text' field with full paper content
        text_content = paper.get('text', '').strip()

        if text_content:
            # Clean the entire paper text first
            cleaned_text = clean_text(text_content)

            # Split long papers into sentences to reach target count
            # Simple sentence splitting on periods followed by space and capital letter
            paper_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_text)

            # Add each sentence as a separate entry
            for sent_idx, sentence in enumerate(paper_sentences):
                # Clean individual sentence
                cleaned_sentence = clean_text(sentence)

                # Validate sentence quality
                if is_valid_sentence(cleaned_sentence):
                    sentence_data = {
                        'text': cleaned_sentence,
                        'source': 'wmdp-bio-retain',
                        'type': 'paper_sentence',
                        'paper_index': idx,
                        'sentence_index': sent_idx
                    }
                    sentences.append(sentence_data)
                else:
                    filtered_count += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} papers, extracted {len(sentences)} sentences so far...")

    print(f"\n✓ Extracted {len(sentences)} total sentences from PubMed papers")
    print(f"  Filtered out {filtered_count} low-quality sentences")

    # Sample to target size if we have more than needed
    if len(sentences) > target_size:
        print(f"Sampling {target_size} sentences from {len(sentences)} available...")
        random.shuffle(sentences)
        sentences = sentences[:target_size]
    elif len(sentences) < target_size:
        print(f"Warning: Only found {len(sentences)} sentences (target: {target_size})")
        print(f"Using all {len(sentences)} available sentences")

    print(f"\n✓ Final dataset size: {len(sentences)} sentences")
    return sentences

def save_jsonl(data: List[Dict], filepath: Path):
    """Save data to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_json(data: List[Dict], filepath: Path):
    """Save to JSON file (MUSE framework format - one JSON per line)"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def create_forget_retain_splits(
    sentences: List[Dict],
    forget_ratios: List[float],
    seed: int,
    output_dir: str
):
    """
    Create forget/retain splits following MUSE paper methodology.

    Args:
        sentences: List of sentence dictionaries
        forget_ratios: List of ratios (e.g., [0.01, 0.05, 0.10, 0.20])
        seed: Random seed for reproducibility
        output_dir: Directory to save output files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_path.absolute()}")

    total_sentences = len(sentences)
    print(f"Total sentences: {total_sentences}")

    # Set random seed
    random.seed(seed)

    # Shuffle data
    shuffled_data = sentences.copy()
    random.shuffle(shuffled_data)

    # Save full dataset in JSON format (MUSE format)
    full_file = output_path / 'wmdp_bio_full.json'
    print(f"\nSaving {full_file}...")
    save_json(shuffled_data, full_file)

    # Also save as JSONL for convenience
    full_jsonl = output_path / 'wmdp_bio_full.jsonl'
    print(f"Saving {full_jsonl}...")
    save_jsonl(shuffled_data, full_jsonl)

    # Create splits for each forget ratio
    for ratio in forget_ratios:
        forget_size = int(total_sentences * ratio)
        forget_name = str(int(ratio * 100)).zfill(2)  # e.g., "10" for 10%

        print(f"\n--- Creating forget{forget_name} split ({ratio*100}% = {forget_size} sentences) ---")

        # Split data
        forget_data = shuffled_data[:forget_size]
        retain_data = shuffled_data[forget_size:]

        # Save forget set
        forget_file = output_path / f'wmdp_bio_forget{forget_name}.json'
        print(f"Saving {forget_file} ({len(forget_data)} sentences)...")
        save_json(forget_data, forget_file)

        # Save retain set
        retain_file = output_path / f'wmdp_bio_retain{forget_name}.json'
        print(f"Saving {retain_file} ({len(retain_data)} sentences)...")
        save_json(retain_data, retain_file)

        # Also save JSONL versions
        forget_jsonl = output_path / f'wmdp_bio_forget{forget_name}.jsonl'
        save_jsonl(forget_data, forget_jsonl)

        retain_jsonl = output_path / f'wmdp_bio_retain{forget_name}.jsonl'
        save_jsonl(retain_data, retain_jsonl)

    print("\n✓ Data splitting complete!")
    print(f"\nAll files saved to: {output_path.absolute()}")

def main():
    """Main function to orchestrate the download and processing"""
    parser = argparse.ArgumentParser(
        description="Download and process WMDP-Bio-Retain corpus for unlearning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python prepare_wmdp_bio_dataset.py
  python prepare_wmdp_bio_dataset.py --target_size 5300 --seed 42
  python prepare_wmdp_bio_dataset.py --forget_ratios 0.10 0.20
        """
    )

    parser.add_argument(
        '--target_size',
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f'Target number of sentences to sample (default: {DEFAULT_TARGET_SIZE})'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_SEED})'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for dataset files (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--forget_ratios',
        type=float,
        nargs='+',
        default=DEFAULT_FORGET_RATIOS,
        help=f'Forget ratios to create (default: {DEFAULT_FORGET_RATIOS})'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WMDP-BIO-RETAIN DATASET PREPARATION")
    print("For Machine Unlearning Experiments")
    print("=" * 80)
    print()

    # Step 1: Download dataset
    dataset = download_wmdp_bio_retain()
    if dataset is None:
        return

    # Step 2: Extract and sample sentences
    sentences = extract_sentences_from_dataset(
        dataset=dataset,
        target_size=args.target_size,
        seed=args.seed
    )

    if not sentences:
        print("Error: No sentences extracted from dataset")
        return

    # Step 3: Create forget/retain splits
    create_forget_retain_splits(
        sentences=sentences,
        forget_ratios=args.forget_ratios,
        seed=args.seed,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()