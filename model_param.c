#include <stdio.h>
#include <stdlib.h>
#include "model_param.h"

// Implémentation des fonctions
// Réactivation et correction de la fonction create_dense_layer
DenseLayer* create_dense_layer(int input_size, int output_size, double* weights, double* biases) {
    DenseLayer* layer = (DenseLayer*)malloc(sizeof(DenseLayer));
    if (!layer) {
        fprintf(stderr, "Erreur d'allocation mémoire pour la couche\n");
        exit(1);
    }
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = weights; // Utilisation directe sans nouvelle allocation
    layer->biases = biases; // Utilisation directe sans nouvelle allocation

    return layer;
}

void forwardDense(DenseLayer *layer, const double *input, double *output) {
    // Parcourir chaque neurone de sortie 'i'
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i]; // Initialiser la sortie du neurone 'i' avec son biais

        // Calculer la somme pondérée des entrées pour le neurone de sortie 'i'
        for (int j = 0; j < layer->input_size; j++) {
            // Index correct pour les poids : j * layer->output_size + i
            // Cela prend le poids correspondant à l'entrée 'j' pour le neurone de sortie 'i'
            output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }

        // Appliquer la fonction d'activation (tanh) à la somme pondérée pour le neurone de sortie 'i'
        output[i] = tanh(output[i]);
    }
}


void softmax(double *input, double *output, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

DenseLayer load_dense_layer_params(const char* weights_path, const char* biases_path, int input_size, int output_size) {
    DenseLayer params;
    params.input_size = input_size;
    params.output_size = output_size;

    params.weights = (double*)malloc(sizeof(double) * input_size * output_size);
    if (!params.weights) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les poids\n");
        exit(1);
    }

    FILE* file = fopen(weights_path, "r");
    if (!file) {
        fprintf(stderr, "Impossible d'ouvrir le fichier de poids %s\n", weights_path);
        exit(1);
    }
    for (int i = 0; i < input_size * output_size; i++) {
        if (fscanf(file, "%lf", &params.weights[i]) != 1) {
            fprintf(stderr, "Erreur de lecture des poids\n");
            exit(1);
        }
    }
    fclose(file);

    params.biases = (double*)malloc(sizeof(double) * output_size);
    if (!params.biases) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        exit(1);
    }

    file = fopen(biases_path, "r");
    if (!file) {
        fprintf(stderr, "Impossible d'ouvrir le fichier de bie %s\n", biases_path);
        exit(1);
    }
    for (int i = 0; i < output_size; i++) {
        if (fscanf(file, "%lf", &params.biases[i]) != 1) {
            fprintf(stderr, "Erreur de lecture des bies\n");
            exit(1);
        }
    }
    fclose(file);

    return params;
}

void free_dense_layer(DenseLayer* layer) {
    if (layer) {
        if (layer->weights) free(layer->weights);
        if (layer->biases) free(layer->biases);
        free(layer);
    }
}