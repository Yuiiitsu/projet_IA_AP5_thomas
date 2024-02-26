import os
import shutil

# Liste des répertoires à traiter
repertoires = ['thomas', 'enzo', 'gauthier', 'nathan', 'matteo']

# Création du répertoire de destination s'il n'existe pas déjà
destination_dir = 'data'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Fonction pour copier et renommer les fichiers
def copier_et_renommer(source_dir, destination_dir, prefixe):
    # Parcours des sous-répertoires
    for sous_repertoire in os.listdir(source_dir):
        sous_repertoire_path = os.path.join(source_dir, sous_repertoire)
        if os.path.isdir(sous_repertoire_path):
            # Parcours des fichiers dans le sous-répertoire
            for root, dirs, files in os.walk(sous_repertoire_path):
                for filename in files:
                    # Construction des chemins complets
                    source_path = os.path.join(root, filename)
                    destination_path = os.path.join(destination_dir, f"{prefixe}_{filename}")
                    # Copie du fichier en le renommant
                    shutil.copy(source_path, destination_path)

# Parcours des répertoires spécifiés
for repertoire in repertoires:
    repertoire_source = repertoire
    copier_et_renommer(repertoire_source, destination_dir, repertoire)
