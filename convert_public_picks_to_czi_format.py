#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour convertir des annotations de particules du format .ndjson vers
le format .json structuré attendu par l'évaluation du CZI Challenge.

Ce script est adapté pour les fichiers d'annotation où les coordonnées sont
déjà en Angstroms et les fichiers sont directement dans un dossier 'Picks'.
Il lit un répertoire de données, trouve les annotations au format .ndjson,
et sauvegarde les fichiers .json résultants dans le même dossier 'Picks'.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

# Correspondance entre les mots-clés des noms de fichiers et les noms officiels des particules CZI.
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
    Lit un fichier .ndjson, extrait les coordonnées des points.
    Les coordonnées sont supposées être déjà en Angstroms.
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
                        # Les coordonnées sont déjà en Angstroms, pas de conversion.
                        points_data.append({'location': location})
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
        print(f"  -> Succès : {len(points_data)} points écrits dans {file_path.name}")
    except Exception as e:
        print(f"  -> ERREUR : Impossible d'écrire dans le fichier {file_path}: {e}")

def process_tomograms(data_dir: str):
    """
    Fonction principale qui parcourt les dossiers, lit les .ndjson et crée les .json.
    """
    data_path = Path(data_dir)

    if not data_path.is_dir():
        print(f"ERREUR : Le répertoire d'entrée n'a pas été trouvé : {data_path}")
        return

    tomo_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("TS")]
    if not tomo_dirs:
        print(f"ERREUR : Aucun sous-dossier de tomogramme (ex: 'TS_*') trouvé dans {data_path}")
        return

    print(f"{len(tomo_dirs)} dossiers de tomogrammes trouvés. Début du traitement...")

    for tomo_dir in sorted(tomo_dirs):
        print(f"\nTraitement du tomogramme : {tomo_dir.name}")
        
        picks_dir = tomo_dir / "Picks"
        if not picks_dir.is_dir():
            print(f"  Aucun dossier 'Picks' trouvé pour {tomo_dir.name}. Ignoré.")
            continue

        for ndjson_file in sorted(picks_dir.glob("*.ndjson")):
            file_stem = ndjson_file.stem.lower()
            
            for keyword, czi_name in KEYWORD_TO_CZI_NAME.items():
                if keyword.lower() in file_stem:
                    print(f"  Fichier d'annotation trouvé : {ndjson_file.name} (type '{czi_name}')")
                    points = parse_ndjson_coordinates(ndjson_file)
                    if points:
                        output_json_path = picks_dir / f"{czi_name}.json"
                        write_czi_json(output_json_path, points)
                    break

def main():
    parser = argparse.ArgumentParser(description="Convertit les annotations .ndjson (coordonnées en Angstroms) en format .json pour le CZI Challenge.")
    parser.add_argument("--data_dir", required=True, help="Chemin vers le dossier racine contenant les sous-dossiers de tomogrammes (ex: /path/to/public_dataset).")
    args = parser.parse_args()
    process_tomograms(args.data_dir)

if __name__ == "__main__":
    main()