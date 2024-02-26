import os
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model('custom_model.keras')

# Créer le dossier pour enregistrer les poids et les biais
os.makedirs("weights", exist_ok=True)
os.makedirs("biaises", exist_ok=True)

for layer in model.layers:
        array = layer.get_weights()
        if len(array) != 0:
            w1,b1 = layer.get_weights()
            np.savetxt("./weights/" + layer.name + '.txt', w1)
            np.savetxt("./biaises/"+ layer.name + '.txt', b1)

print("Extraction des poids et biais terminée.")





















# # Parcourir toutes les couches du modèle
# for i, layer in enumerate(model.layers):
#     # Vérifier si la couche a des poids
#     if layer.weights:
#         # Extraire les poids et les biais de la couche
#         weights = layer.get_weights()[0]
#         biases = layer.get_weights()[1]
        
#         # Enregistrer les biais dans un fichier
#         np.savetxt(f"weights_and_biases/layer_{i}_biases.txt", biases.reshape(-1, 1), delimiter=",", fmt='%1.4f')

#         # Enregistrer les poids dans un fichier
#         np.savetxt(f"weights_and_biases/layer_{i}_weights.txt", weights, delimiter=",", fmt='%1.4f')
        

# print("Poids et biais extraits et enregistrés avec succès.")