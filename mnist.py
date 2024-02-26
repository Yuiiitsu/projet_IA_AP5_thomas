import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import subprocess
import sys
import random 
from tensorflow.keras.preprocessing import image


import itertools
# Définition des répertoires de l'ensemble d'entraînement et de l'ensemble de test
thomas_base_dir = "data"
jordan_test_dir = "thomas"

# Taille des images
image_size = (28, 28)

# Taille du batch
batch_size = 1

# Générateur de données pour l'ensemble d'entraînement
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    thomas_base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale')

# Générateur de données pour l'ensemble de test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    jordan_test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale')


# Construction du modèle avec MLP
model = models.Sequential([
    layers.Flatten(input_shape=(image_size[0], image_size[1], 1)),  
    layers.Dense(512, activation='tanh'),
    layers.Dense(512, activation='tanh'),
    layers.Dense(10, activation='softmax') 
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    epochs=30,
    validation_data=test_generator,
    validation_steps=max(1, test_generator.samples // batch_size))

# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Précision sur le jeu de test: {test_accuracy:.4f}')

# Sauvegarde du modèle
model.save('custom_model.keras')
print('Modèle sauvegardé sous custom_model.keras')

# Tracé des courbes de perte et de précision
plt.figure(figsize=(10, 5))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perte Entraînement')
plt.plot(history.history['val_loss'], label='Perte Validation')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.title('Perte Entraînement et Validation')
plt.legend()

# Courbe de précision
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Précision Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision Validation')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.title('Précision Entraînement et Validation')
plt.legend()

# Sauvegarde des graphiques
plt.savefig('loss_accuracy_plot.png')

# Tracé du modèle
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Affichage d'une grille d'images avec les prédictions et les vraies étiquettes
num_images_to_show = 100
test_images, test_labels = [], []

# Récupération d'un nombre suffisant d'images et d'étiquettes
for _ in range(num_images_to_show):
    img, label = next(test_generator)
    test_images.append(img[0])  # Ajout de la première image du lot
    test_labels.append(label[0])  # Ajout de la première étiquette du lot

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Prédictions pour ces images
predictions = model.predict(test_images)

# Affichage d'une grille d'images avec les prédictions, les vraies étiquettes et les probabilités
num_rows = 10
num_cols = 10
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

for i, ax in enumerate(axes.flat):
    img = test_images[i]
    ax.imshow(img.squeeze(), cmap='gray')  # Assurez-vous que img.squeeze() a la bonne forme
    pred_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])
    confidence = predictions[i][pred_label]  # Probabilité associée à la classe prédite
    print(f'Pred: {pred_label}, True: {true_label}, Confidence: {confidence:.2f}')
    ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}\nConfidence: {confidence:.2f}', color='green' if pred_label == true_label else 'red')
    ax.axis('off')

plt.tight_layout()
plt.savefig('predictions_grid_with_confidence.png')







import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers

def build_model(neurons_per_layer, activation):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(image_size[0], image_size[1], 1)))
    for n_neurons in neurons_per_layer:
        model.add(layers.Dense(n_neurons, activation=activation))
    model.add(layers.Dense(10, activation='softmax'))  # Ajustez le nombre de classes si nécessaire
    return model

# Options pour les configurations du modèle
n_layers_options = [1, 2, 3]
n_neurons_options = [128, 256, 512]
activation_options = ['relu', 'tanh', 'sigmoid']
epochs = 30
repeats = 5

model_performance = {}

# Générer toutes les combinaisons possibles de nombres de neurones pour chaque nombre de couches
neurons_combinations = []
for n_layers in n_layers_options:
    if n_layers == 1:
        for n in n_neurons_options:
            neurons_combinations.append([n])
    else:
        # Générer des combinaisons avec la contrainte que chaque couche suivante ait un nombre de neurones inférieur ou égal à la couche précédente
        for combination in itertools.product(n_neurons_options, repeat=n_layers):
            if all(combination[i] >= combination[i + 1] for i in range(len(combination) - 1)):
                neurons_combinations.append(list(combination))


# Itération sur toutes les combinaisons de configurations
for neurons_per_layer, activation in itertools.product(neurons_combinations, activation_options):
    accuracies, losses = [], []
    for _ in range(repeats):
        current_model = build_model(neurons_per_layer, activation)
        current_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = current_model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // batch_size),
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=max(1, test_generator.samples // batch_size))
        
        test_loss, test_accuracy = current_model.evaluate(test_generator)
        accuracies.append(test_accuracy)
        losses.append(test_loss)
    
    # Calculer la moyenne des précisions et des pertes pour la configuration actuelle
    average_accuracy = np.mean(accuracies)
    average_loss = np.mean(losses)
    
    model_performance[f'{len(neurons_per_layer)} layers, {neurons_per_layer} neurons, {activation}'] = (average_loss, average_accuracy)

# Préparation des données pour les graphiques
configurations = list(model_performance.keys())
average_losses = [performance[0] for performance in model_performance.values()]
average_accuracies = [performance[1] for performance in model_performance.values()]

# Graphique pour les pertes
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(configurations, average_losses, s=np.array(average_losses) * 1000, alpha=0.5)
plt.xticks(rotation=90)
plt.ylabel('Moyenne des pertes')
plt.title('Pertes moyennes par configuration')

# Graphique pour les précisions
plt.subplot(1, 2, 2)
plt.scatter(configurations, average_accuracies, s=np.array(average_accuracies) * 1000, alpha=0.5)
plt.xticks(rotation=90)
plt.ylabel('Moyenne des précisions')
plt.title('Précisions moyennes par configuration')

plt.tight_layout()
plt.savefig('model_performance_comparison.png')


# Sauvegarde du modèle avec les meilleures performances
best_model_config = configurations[np.argmax(average_accuracies)]
best_model = build_model(neurons_per_layer, activation)
best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
best_model.save('best_model.keras')
print(f'Meilleur modèle sauvegardé sous best_model.keras : {best_model_config}')




#python_executable = sys.executable
#subprocess.run([python_executable, 'extract_weights_and_biases.py'])













# # Chemin vers le répertoire de test
# test_dir = "thomas"

# # Sélection aléatoire d'un sous-répertoire (classe)
# random_class = random.choice(os.listdir(test_dir))
# class_dir = os.path.join(test_dir, random_class)

# # Sélection aléatoire d'une image dans le sous-répertoire
# random_image = random.choice(os.listdir(class_dir))
# test_image_path = os.path.join(class_dir, random_image)

# # Charger l'image de test
# img = image.load_img(test_image_path, target_size=image_size)

# # Prétraitement de l'image
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.  # Normalisation des valeurs de pixel

# # Faire la prédiction
# prediction = model.predict(img_array)
# predicted_class = np.argmax(prediction)
# confidence = prediction[0][predicted_class]

# # Afficher la prédiction et la confiance associée
# print(f"Prédiction : {predicted_class}, Confiance : {confidence:.2f}")