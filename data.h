#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Function prototypes
void generate_synthetic_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim, 
                           float input_min, float input_max);
void save_data(float* X, float* y, int num_sequences, int seq_len, int input_dim, int output_dim, const char* filename);

#endif
