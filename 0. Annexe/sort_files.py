import os
import shutil
import sys

# Vérifier si un argument (chemin du répertoire) est fourni
if len(sys.argv) > 1:
    # Utiliser le répertoire fourni comme argument
    current_directory = sys.argv[1]
else:
    # Sinon, utiliser le répertoire courant
    current_directory = os.getcwd()

# Créer des répertoires de 0 à 9 s'ils n'existent pas
for i in range(10):
    dir_path = os.path.join(current_directory, str(i))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Parcourir tous les fichiers et répertoires du répertoire spécifié
for filename in os.listdir(current_directory):
    source_path = os.path.join(current_directory, filename)
    # Vérifier si le chemin est un fichier et si le premier caractère est un chiffre de 0 à 9
    if os.path.isfile(source_path) and filename[0].isdigit():
        # Construire le chemin de destination basé sur le premier caractère
        destination_directory = os.path.join(current_directory, filename[0])
        destination_path = os.path.join(destination_directory, filename)

        # Déplacer le fichier dans le répertoire correspondant
        shutil.move(source_path, destination_path)

print("Triage terminé.")
