#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert particle annotations from .ndjson format to
the structured .json format expected by the CZI Challenge evaluation.

This script reads an evaluation data directory, finds the raw annotations
in .ndjson format, converts them to Angstroms, and saves the
resulting .json files directly into the existing folder structure,
under `tomo_name/Picks/particle_name.json`.
"""

import argparse
import os
import json
from pathlib import Path
from collections import defaultdict

# Mapping between filename keywords and official CZI particle names.
# The keyword 'vlp' is used for 'virus-like-particle' and 'ferritin' for 'apo-ferritin'.
KEYWORD_TO_CZI_NAME = {
    'ferritin': 'apo-ferritin',
    'beta_amylase': 'beta-amylase',
    'beta_galactosidase': 'beta-galactosidase',
    'ribosome': 'ribosome',
    'thyroglobulin': 'thyroglobulin',
    'vlp': 'virus-like-particle',
}

def parse_ndjson_coordinates(file_path: Path, voxel_spacing: float) -> list:
    """
    Reads a .ndjson file, extracts point coordinates, and converts them to Angstroms.
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
                        # NEW: Convert pixel coordinates to Angstroms
                        angstrom_location = {
                            'x': location['x'] * voxel_spacing,
                            'y': location['y'] * voxel_spacing,
                            'z': location['z'] * voxel_spacing,
                        }
                        points_data.append({'location': angstrom_location})
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
        print(f"  -> Success: {len(points_data)} points written to {file_path}")
    except Exception as e:
        print(f"  -> ERROR: Could not write to file {file_path}: {e}")

def process_tomograms(input_dir: str, output_dir: str):
    """
    Main function that traverses directories, reads .ndjson files, and creates .json files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"ERROR: Input directory not found: {input_path}")
        return

    # We look for all subdirectories in the input directory.
    # We assume that each subdirectory corresponds to a tomogram.
    tomo_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("TS")]

    if not tomo_dirs:
        print(f"ERROR: No subdirectories found in {input_path}")
        return

    print(f"{len(tomo_dirs)} tomogram directories found. Starting processing...")

    for tomo_dir in sorted(tomo_dirs):
        print(f"\nProcessing tomogram: {tomo_dir.name}")
        
        particle_points = defaultdict(list)

        # The search pattern is updated to match the new structure:
        # tomogram_name/Reconstructions/VoxelSpacing*/Annotations/session_*/... .ndjson
        search_pattern = "Reconstructions/VoxelSpacing*/Annotations/*/*.ndjson"
        for ndjson_file in sorted(tomo_dir.glob(search_pattern)):
            # --- NEW VOXEL SPACING EXTRACTION LOGIC ---
            voxel_spacing = None
            try:
                # We look for a parent in the path that starts with "VoxelSpacing"
                for part in ndjson_file.parts:
                    if part.startswith("VoxelSpacing"):
                        # Extract the numeric value. Ex: "VoxelSpacing4.990" -> 4.990
                        voxel_spacing = float(part.replace("VoxelSpacing", ""))
                        break
            except (ValueError, IndexError):
                pass # Handle the error below

            if voxel_spacing is None:
                print(f"  Warning: Voxel spacing not found in the path of {ndjson_file}. File ignored.")
                continue

            # --- NEW VERIFICATION LOGIC ---
            # We look for a .json file in the same directory as the .ndjson
            session_dir = ndjson_file.parent
            metadata_files = list(session_dir.glob('*.json'))

            is_valid_gt = True # By default, we consider the data valid
            if metadata_files:
                metadata_path = metadata_files[0]
                if len(metadata_files) > 1:
                    print(f"  Warning: Multiple .json files found in {session_dir}. Using the first one: {metadata_path.name}")
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if metadata.get('ground_truth_status') is False:
                            is_valid_gt = False
                            print(f"  -> Ignored: 'ground_truth_status' is False in {metadata_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not read or parse metadata file {metadata_path.name}: {e}")

            if not is_valid_gt:
                continue # Move to the next .ndjson file
            
            file_stem = ndjson_file.stem.lower()
            
            for keyword, czi_name in KEYWORD_TO_CZI_NAME.items():
                if keyword.lower() in file_stem:
                    print(f"  Valid file found: {ndjson_file.relative_to(tomo_dir)} (type '{czi_name}', VS={voxel_spacing:.3f})")
                    points = parse_ndjson_coordinates(ndjson_file, voxel_spacing)
                    if points:
                        particle_points[czi_name].extend(points)
                    break

        if not particle_points:
            print(f"  No valid particle data found for tomogram {tomo_dir.name}.")
            continue

        picks_dir = output_path / tomo_dir.name / "Picks"
        for czi_name, points_data in particle_points.items():
            output_json_path = picks_dir / f"{czi_name}.json"
            write_czi_json(output_json_path, points_data)

# Example usage:
# python prepare_public_eval_annotations.py --data_dir /path/to/eval/data

def main():
    parser = argparse.ArgumentParser(description="Converts .ndjson annotations to .json format for the CZI Challenge.")
    parser.add_argument("--data_dir", required=True, help="Path to the root directory containing tomogram subdirectories. The .json files will be created inside this structure.")
    args = parser.parse_args()
    process_tomograms(args.data_dir, args.data_dir)

if __name__ == "__main__":
    main()
