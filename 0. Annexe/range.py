import os
import shutil

# Chemin du répertoire contenant les images
repertoire_images = "data"

# Créer les répertoires de 0 à 9 s'ils n'existent pas déjà
for i in range(10):
    nom_repertoire = str(i)
    chemin_repertoire = os.path.join(repertoire_images, nom_repertoire)
    if not os.path.exists(chemin_repertoire):
        os.makedirs(chemin_repertoire)

# Parcourir les fichiers dans le répertoire d'images
for fichier in os.listdir(repertoire_images):
    if fichier.endswith(".bmp"):
        # Extraire le numéro du fichier
        numero = fichier.split("_")[1]
        # Déplacer le fichier vers le répertoire correspondant au numéro
        chemin_source = os.path.join(repertoire_images, fichier)
        chemin_destination = os.path.join(repertoire_images, numero, fichier)
        shutil.move(chemin_source, chemin_destination)

print("Le rangement des images est terminé.")
