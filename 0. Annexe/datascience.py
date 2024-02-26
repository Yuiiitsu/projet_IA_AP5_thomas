

def build_model(n_layers, n_neurons, activation):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(image_size[0], image_size[1], 1)))
    for _ in range(n_layers):
        model.add(layers.Dense(n_neurons, activation=activation))
    model.add(layers.Dense(10, activation='softmax'))  # Ajustez le nombre de classes si nécessaire
    return model

# Options pour les configurations du modèle
n_layers_options = [1, 2, 3]  # Exemple : 1, 2, ou 3 couches cachées
n_neurons_options = [128, 256, 512]  # Exemple : 128, 256, ou 512 neurones par couche
activation_options = ['relu', 'tanh', 'sigmoid']  # Fonctions d'activation à tester
epochs = 30  # Limite d'époques pour l'entraînement

model_performance = {}

# Itération sur toutes les combinaisons de configurations
for n_layers, n_neurons, activation in itertools.product(n_layers_options, n_neurons_options, activation_options):
    # Utiliser une autre variable pour stocker le modèle testé
    current_model = build_model(n_layers, n_neurons, activation)
    current_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Entraînement du modèle avec {n_layers} couches, {n_neurons} neurones, activation {activation}")
    history = current_model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=max(1, test_generator.samples // batch_size))
    
    test_loss, test_accuracy = current_model.evaluate(test_generator)
    print(f'Précision sur le jeu de test: {test_accuracy:.4f} - Paramètres: Couches={n_layers}, Neurones={n_neurons}, Activation={activation}')
    
    # Enregistrer la précision du modèle avec ses paramètres comme clé
    model_performance[f'{n_layers} layers, {n_neurons} neurons, {activation}'] = test_accuracy

# Tracé des performances des modèles
plt.figure(figsize=(10, 8))
plt.barh(list(model_performance.keys()), list(model_performance.values()))
plt.xlabel('Précision sur le jeu de test')
plt.ylabel('Configuration du modèle')
plt.xlim(0.5, 1.0)

plt.title('Précision des différents modèles')
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig('model_performance_comparison.png')