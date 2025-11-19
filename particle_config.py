# Fichier de configuration pour définir les propriétés des particules
# pour la génération des données d'entraînement YOLO.


# 3. Configuration pour l'évaluation de type CZI Challenge.
#    Les rayons sont en Angstroms. Les poids sont utilisés pour le score final.
CZI_PARTICLE_CONFIG = {
    'apo-ferritin':       {'radius': 60, 'weight': 1.0, 'id': 1},
    'beta-amylase':       {'radius': 65, 'weight': 0.0, 'id': 2},  # Poids 0 car non évalué.
    'beta-galactosidase': {'radius': 90, 'weight': 2.0, 'id': 3},   
    'ribosome':           {'radius': 150, 'weight': 1.0, 'id': 4},
    'thyroglobulin':      {'radius': 130, 'weight': 2.0, 'id': 5},
    'virus-like-particle':{'radius': 135, 'weight': 1.0, 'id': 6},
}