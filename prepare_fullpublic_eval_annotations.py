#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour convertir des annotations de particules du format .ndjson vers
le format .json structuré attendu par l'évaluation du CZI Challenge.

Ce script lit un répertoire de données d'évaluation, trouve les annotations
brutes au format .ndjson, les convertit en Angstroms, et sauvegarde les
fichiers .json résultants directement dans la structure de dossiers existante,
sous `nom_tomo/Picks/nom_particule.json`.
"""

import argparse
import os
import json
from pathlib import Path
from collections import defaultdict

# Correspondance entre les mots-clés des noms de fichiers et les noms officiels des particules CZI.
# Le mot-clé 'vlp' est utilisé pour 'virus-like-particle' et 'ferritin' pour 'apo-ferritin'.
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
    Lit un fichier .ndjson, extrait les coordonnées des points et les convertit en Angstroms.
    Retourne une liste de dictionnaires au format CZI.
    """
    points_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('type') == 'point':
                    location = data.get('location')
                    if location and all(k in location for k in ['x', 'y', 'z']):
                        # NOUVEAU: Convertir les coordonnées pixel en Angstroms
                        angstrom_location = {
                            'x': location['x'] * voxel_spacing,
                            'y': location['y'] * voxel_spacing,
                            'z': location['z'] * voxel_spacing,
                        }
                        points_data.append({'location': angstrom_location})
    except Exception as e:
        print(f"Avertissement : Impossible de traiter le fichier {file_path}: {e}")
    return points_data

def write_czi_json(file_path: Path, points_data: list):
    """
    Écrit les données des points dans un fichier JSON au format attendu par l'évaluation CZI.
    """
    # La structure finale du JSON est {"points": [ { "location": ... }, ... ]}
    output_data = {'points': points_data}
    try:
        # S'assurer que le dossier parent existe
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"  -> Succès : {len(points_data)} points écrits dans {file_path}")
    except Exception as e:
        print(f"  -> ERREUR : Impossible d'écrire dans le fichier {file_path}: {e}")

def process_tomograms(input_dir: str, output_dir: str):
    """
    Fonction principale qui parcourt les dossiers, lit les .ndjson et crée les .json.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"ERREUR : Le répertoire d'entrée n'a pas été trouvé : {input_path}")
        return

    # On cherche tous les sous-dossiers dans le répertoire d'entrée.
    # On suppose que chaque sous-dossier correspond à un tomogramme.
    tomo_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("TS")]

    if not tomo_dirs:
        print(f"ERREUR : Aucun sous-dossier trouvé dans {input_path}")
        return

    print(f"{len(tomo_dirs)} dossiers de tomogrammes trouvés. Début du traitement...")

    for tomo_dir in sorted(tomo_dirs):
        print(f"\nTraitement du tomogramme : {tomo_dir.name}")
        
        particle_points = defaultdict(list)

        # Le pattern de recherche est mis à jour pour correspondre à la nouvelle structure :
        # nom_tomogram/Reconstructions/VoxelSpacing*/Annotations/session_*/... .ndjson
        search_pattern = "Reconstructions/VoxelSpacing*/Annotations/*/*.ndjson"
        for ndjson_file in sorted(tomo_dir.glob(search_pattern)):
            # --- NOUVELLE LOGIQUE D'EXTRACTION DU VOXEL SPACING ---
            voxel_spacing = None
            try:
                # On cherche un parent du chemin qui commence par "VoxelSpacing"
                for part in ndjson_file.parts:
                    if part.startswith("VoxelSpacing"):
                        # Extraire la valeur numérique. Ex: "VoxelSpacing4.990" -> 4.990
                        voxel_spacing = float(part.replace("VoxelSpacing", ""))
                        break
            except (ValueError, IndexError):
                pass # Gérer l'erreur plus bas

            if voxel_spacing is None:
                print(f"  Avertissement : Voxel spacing non trouvé dans le chemin de {ndjson_file}. Fichier ignoré.")
                continue

            # --- NOUVELLE LOGIQUE DE VÉRIFICATION ---
            # On cherche un fichier .json dans le même dossier que le .ndjson
            session_dir = ndjson_file.parent
            metadata_files = list(session_dir.glob('*.json'))

            is_valid_gt = True # Par défaut, on considère les données valides
            if metadata_files:
                metadata_path = metadata_files[0]
                if len(metadata_files) > 1:
                    print(f"  Avertissement : Plusieurs fichiers .json trouvés dans {session_dir}. Utilisation du premier : {metadata_path.name}")
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if metadata.get('ground_truth_status') is False:
                            is_valid_gt = False
                            print(f"  -> Ignoré : 'ground_truth_status' est à False dans {metadata_path.name}")
                except Exception as e:
                    print(f"  Avertissement : Impossible de lire ou parser le fichier de métadonnées {metadata_path.name}: {e}")

            if not is_valid_gt:
                continue # Passer au fichier .ndjson suivant
            
            file_stem = ndjson_file.stem.lower()
            
            for keyword, czi_name in KEYWORD_TO_CZI_NAME.items():
                if keyword.lower() in file_stem:
                    print(f"  Fichier valide trouvé : {ndjson_file.relative_to(tomo_dir)} (type '{czi_name}', VS={voxel_spacing:.3f})")
                    points = parse_ndjson_coordinates(ndjson_file, voxel_spacing)
                    if points:
                        particle_points[czi_name].extend(points)
                    break

        if not particle_points:
            print(f"  Aucune donnée de particule valide trouvée pour le tomogramme {tomo_dir.name}.")
            continue

        picks_dir = output_path / tomo_dir.name / "Picks"
        for czi_name, points_data in particle_points.items():
            output_json_path = picks_dir / f"{czi_name}.json"
            write_czi_json(output_json_path, points_data)

# Exemple d'utilisation :
# python prepare_public_eval_annotations.py --data_dir /chemin/vers/donnees/eval

def main():
    parser = argparse.ArgumentParser(description="Convertit les annotations .ndjson en format .json pour le CZI Challenge.")
    parser.add_argument("--data_dir", required=True, help="Chemin vers le dossier racine contenant les sous-dossiers de tomogrammes. Les fichiers .json seront créés à l'intérieur de cette structure.")
    args = parser.parse_args()
    process_tomograms(args.data_dir, args.data_dir)

if __name__ == "__main__":
    main()
