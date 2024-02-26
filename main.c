#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model_param.h"
#include "Bmp2Matrix.h"
#include "model_param.c"
#include "Bmp2Matrix.c"

int main() {
    BMP bitmap;
    FILE* pFichier = NULL;

    pFichier = fopen("data/9/gauthier_9_0.bmp", "rb"); // Ouverture du fichier BMP
    if (pFichier == NULL) {
        fprintf(stderr, "Erreur dans la lecture du fichier\n");
        exit(1);
    }

    LireBitmap(pFichier, &bitmap);
    fclose(pFichier); // Fermeture du fichier

    if (bitmap.infoHeader.largeur != 28 || bitmap.infoHeader.hauteur != 28) {
        fprintf(stderr, "L'image doit être de 28x28 pixels\n");
        DesallouerBMP(&bitmap);
        exit(1);
    }

    ConvertRGB2Gray(&bitmap);

    double input[784];
    for (int i = 0; i < 784; i++) {
        // Conversion des données de l'image en entrée normalisée pour le réseau de neurones
        // Supposons que bitmap.mPixelsGray est un tableau 2D de 28x28
        input[i] = bitmap.mPixelsGray[i / 28][i % 28];
    }

    DesallouerBMP(&bitmap);

    DenseLayer layer1_params = load_dense_layer_params("weights/dense.txt", "biaises/dense.txt", 784, 512);
    DenseLayer layer2_params = load_dense_layer_params("weights/dense_1.txt", "biaises/dense_1.txt", 512, 512);
    DenseLayer layer3_params = load_dense_layer_params("weights/dense_2.txt", "biaises/dense_2.txt", 512, 10);

    DenseLayer* layer1 = create_dense_layer(784, 512, layer1_params.weights, layer1_params.biases);
    DenseLayer* layer2 = create_dense_layer(512, 512, layer2_params.weights, layer2_params.biases);
    DenseLayer* layer3 = create_dense_layer(512, 10, layer3_params.weights, layer3_params.biases);

    double* output1 = (double*)malloc(sizeof(double) * 512);
    double* output2 = (double*)malloc(sizeof(double) * 512);
    double* finalOutput = (double*)malloc(sizeof(double) * 10);

    printf("Entrées du réseau :\n");
    for (int i = 0; i < 784; i++) {
        printf("%f ", input[i]);
        if ((i + 1) % 28 == 0) printf("\n"); // Nouvelle ligne tous les 28 éléments pour une meilleure lisibilité
    }
    printf("\n");


    forwardDense(layer1, input, output1);
    forwardDense(layer2, output1, output2);
    softmax(output2, finalOutput, 10);

    printf("Probabilités de sortie (après softmax) :\n");
    for (int i = 0; i < 10; i++) {
        printf("Classe %d: %f\n", i, finalOutput[i]);
    }



    free(output1);
    free(output2);
    free(finalOutput);
    // Libération correcte des couches et des structures
    free_dense_layer(layer1);
    free_dense_layer(layer2);
    free_dense_layer(layer3);

    return 0;
}