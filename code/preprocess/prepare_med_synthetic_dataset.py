"""
This code generates forget/retain splits from a synthetic medical dataset
"""

import argparse
import json
import os
import uuid
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def make_client_names_individually_unique_in_memory(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a new list of records where repeated client_name values
    after the first occurrence get a short uuid suffix appended.

    Example: first `Alice` stays `Alice`, second becomes `Alice-1a2b3c4d`, etc.
    """
    name_count: Dict[str, int] = {}
    out_records: List[Dict[str, Any]] = []

    for rec in records:
        # Copy record shallowly to avoid mutating original (defensive)
        record = dict(rec)
        client_name = record.get('client_name')

        if client_name is None:
            out_records.append(record)
            continue

        current_count = name_count.get(client_name, 0)
        if current_count == 0:
            # first occurrence: keep as-is
            name_count[client_name] = 1
        else:
            # subsequent occurrences: add short uuid suffix
            unique_id = uuid.uuid4().hex[:8]
            record['client_name'] = f"{client_name}-{unique_id}"
            name_count[client_name] = current_count + 1

        out_records.append(record)

    return out_records


def save_json_lines(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save a list of dicts as newline-delimited JSON (JSONL).
    We keep the same behavior as previous scripts which used one JSON
    object per line (even when naming the file .json).
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def create_forget_retain_splits_from_records(records: List[Dict[str, Any]], forget_ratios=[0.10], seed=42, output_dir='dataset', dry_run: bool = False, sample: int = 3):
    """Create and save forget/retain splits to output_dir.

    records: list of dicts (already processed / unique-named)
    forget_ratios: list of ratios (e.g., [0.01, 0.05, 0.10])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_records = len(records)
    random.seed(seed)

    # Shuffle a copy
    shuffled = list(records)
    random.shuffle(shuffled)

    # Save the full dataset into output_dir (line-delimited JSON)
    # Use med_synthetic_ prefix for output filenames
    full_file = output_path / 'med_synthetic_full.json'
    if dry_run:
        print(f"[dry-run] Would save full dataset ({total_records} records) to: {full_file}")
        # show a small sample
        print(f"[dry-run] Sample records ({min(sample, total_records)}):")
        for i, r in enumerate(shuffled[:sample], 1):
            print(f"  {i}. {json.dumps(r)}")
    else:
        save_json_lines(shuffled, str(full_file))
        print(f"Saved full dataset ({total_records} records) to: {full_file}")

    for ratio in forget_ratios:
        forget_size = int(total_records * ratio)
        forget_name = str(int(ratio * 100)).zfill(2)

        forget_data = shuffled[:forget_size]
        retain_data = shuffled[forget_size:]

        forget_file = output_path / f'med_synthetic_forget{forget_name}.json'
        retain_file = output_path / f'med_synthetic_full_minus_forget{forget_name}.json'

        if dry_run:
            print(f"[dry-run] Would save forget set {forget_file} ({len(forget_data)} records)")
            print(f"[dry-run] Would save retain set {retain_file} ({len(retain_data)} records)")
        else:
            save_json_lines(forget_data, str(forget_file))
            save_json_lines(retain_data, str(retain_file))
            print(f"Saved forget set {forget_file} ({len(forget_data)} records)")
            print(f"Saved retain set {retain_file} ({len(retain_data)} records)")

    print('\nAll outputs written to:', output_path.resolve())


def prepare_medical_synthetic_dataset(input_file: str = 'dataset/synthetic_soap_notes.jsonl', forget_ratios=[0.10, 0.01, 0.05, 0.20], seed: int = 42, output_dir: str = 'dataset', dry_run: bool = False, sample: int = 3):
    """High-level convenience function to run the full pipeline:
    - load JSONL
    - make client names unique (in memory)
    - create and save splits into `output_dir`

    Note: does not write any intermediate files to project root.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading input file: {input_file}")
    records = load_jsonl(input_file)
    print(f"Loaded {len(records)} records")

    print('Making client names individually unique (in memory)...')
    unique_records = make_client_names_individually_unique_in_memory(records)

    print('Creating forget/retain splits' + (' (dry-run, not writing files)...' if dry_run else ' and writing outputs...'))
    create_forget_retain_splits_from_records(unique_records, forget_ratios=forget_ratios, seed=seed, output_dir=output_dir, dry_run=dry_run, sample=sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare medical synthetic dataset (make names unique + splits).')
    parser.add_argument('--input-file', '-i', default='dataset/synthetic_soap_notes.jsonl', help='Path to input JSONL')
    parser.add_argument('--output-dir', '-o', default='dataset', help='Output directory for splits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Do not write files; show what would be written and print sample records')
    parser.add_argument('--sample', type=int, default=3, help='Number of sample records to show in dry-run')
    parser.add_argument('--forget-ratios', nargs='*', type=float, default=[0.10, 0.01, 0.05, 0.20], help='List of forget ratios (e.g., 0.10 0.01)')

    args = parser.parse_args()

    prepare_medical_synthetic_dataset(input_file=args.input_file, forget_ratios=args.forget_ratios, seed=args.seed, output_dir=args.output_dir, dry_run=args.dry_run, sample=args.sample)
