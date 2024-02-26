#ifndef MODEL_PARAM_H
#define MODEL_PARAM_H

typedef struct {
    int input_size;
    int output_size;
    double* weights;
    double* biases; 
} DenseLayer;

DenseLayer* create_dense_layer(int input_size, int output_size, double* weights, double* biases);
void free_dense_layer(DenseLayer* layer);
DenseLayer load_dense_layer_params(const char* weights_path, const char* biases_path, int input_size, int output_sizevoid);
void softmax(double *input, double *output, int length);

#endif /* MODEL_PARAM_H */
