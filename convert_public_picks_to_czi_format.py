#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert particle annotations from .ndjson format to
the structured .json format expected by the CZI Challenge evaluation.

This script is adapted for annotation files where the coordinates are
already in Angstroms and the files are directly in a 'Picks' folder.
It reads a data directory, finds the annotations in .ndjson format,
and saves the resulting .json files in the same 'Picks' folder.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

# Mapping between filename keywords and official CZI particle names.
KEYWORD_TO_CZI_NAME = {
    'ferritin': 'apo-ferritin',
    'beta_amylase': 'beta-amylase',
    'beta_galactosidase': 'beta-galactosidase',
    'ribosome': 'ribosome',
    'thyroglobulin': 'thyroglobulin',
    'vlp': 'virus-like-particle',
}

def parse_ndjson_coordinates(file_path: Path) -> list:
    """
    Reads a .ndjson file, extracts point coordinates.
    Coordinates are assumed to be already in Angstroms.
    Returns a list of dictionaries in CZI format.
    """
    points_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('type') == 'point':
                    location = data.get('location')
                    if location and all(k in location for k in ['x', 'y', 'z']):
                        # Coordinates are already in Angstroms, no conversion needed.
                        points_data.append({'location': location})
    except Exception as e:
        print(f"Warning: Could not process file {file_path}: {e}")
    return points_data

def write_czi_json(file_path: Path, points_data: list):
    """
    Writes point data to a JSON file in the format expected by CZI evaluation.
    """
    # The final JSON structure is {"points": [ { "location": ... }, ... ]}
    output_data = {'points': points_data}
    try:
        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"  -> Success: {len(points_data)} points written to {file_path.name}")
    except Exception as e:
        print(f"  -> ERROR: Could not write to file {file_path}: {e}")

def process_tomograms(data_dir: str):
    """
    Main function that traverses directories, reads .ndjson files, and creates .json files.
    """
    data_path = Path(data_dir)

    if not data_path.is_dir():
        print(f"ERROR: Input directory not found: {data_path}")
        return

    tomo_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("TS")]
    if not tomo_dirs:
        print(f"ERROR: No tomogram subdirectories (e.g., 'TS_*') found in {data_path}")
        return

    print(f"{len(tomo_dirs)} tomogram directories found. Starting processing...")

    for tomo_dir in sorted(tomo_dirs):
        print(f"\nProcessing tomogram: {tomo_dir.name}")
        
        picks_dir = tomo_dir / "Picks"
        if not picks_dir.is_dir():
            print(f"  No 'Picks' directory found for {tomo_dir.name}. Skipped.")
            continue

        for ndjson_file in sorted(picks_dir.glob("*.ndjson")):
            file_stem = ndjson_file.stem.lower()
            
            for keyword, czi_name in KEYWORD_TO_CZI_NAME.items():
                if keyword.lower() in file_stem:
                    print(f"  Annotation file found: {ndjson_file.name} (type '{czi_name}')")
                    points = parse_ndjson_coordinates(ndjson_file)
                    if points:
                        output_json_path = picks_dir / f"{czi_name}.json"
                        write_czi_json(output_json_path, points)
                    break

def main():
    parser = argparse.ArgumentParser(description="Converts .ndjson annotations (coordinates in Angstroms) to .json format for the CZI Challenge.")
    parser.add_argument("--data_dir", required=True, help="Path to the root directory containing tomogram subdirectories (e.g., /path/to/public_dataset).")
    args = parser.parse_args()
    process_tomograms(args.data_dir)

if __name__ == "__main__":
    main()