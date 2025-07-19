#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SYNTHETIC_OUTPUTS 4
#define INPUT_RANGE_MIN -3.0f
#define INPUT_RANGE_MAX 3.0f

// Function prototypes
float synth_fn(const float* x, int fx, int dim);

void generate_synthetic_sequence_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim);
void generate_linear_sequence_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim);

void save_synthetic_sequence_data_to_csv(float* X, float* y, int num_sequences, int seq_len, int input_dim, int output_dim, const char* filename);
void load_sequence_csv(const char* filename, float** X, float** y, int* num_sequences, int* seq_len, int input_dim, int output_dim);

void reshape_data_for_batch_processing(float* X, float* y, 
                                     float** X_reshaped, float** y_reshaped,
                                     int num_sequences, int seq_len, 
                                     int input_dim, int output_dim);

#endif // DATA_H