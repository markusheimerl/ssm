#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SYNTHETIC_OUTPUTS 4
#define INPUT_RANGE_MIN -3.0f
#define INPUT_RANGE_MAX 3.0f

float synth_fn(const float* x, int sample_idx, int fx, int dim) {
    float t = sample_idx * 0.1;
    
    switch(dim % MAX_SYNTHETIC_OUTPUTS) {
        case 0: 
            return sinf(x[0 % fx]*2)*cosf(x[1 % fx]*1.5f) * expf(-0.1f * t) + 
                   powf(x[2 % fx],2)*x[3 % fx] + 
                   expf(-powf(x[4 % fx]-x[5 % fx],2)) + 
                   0.5f*sinf(x[6 % fx]*x[7 % fx]*(float)M_PI) * (1.0f - expf(-0.2f * t)) +
                   tanhf(x[8 % fx] + x[9 % fx]) +
                   0.3f*cosf(x[10 % fx]*x[11 % fx]) +
                   0.2f*powf(x[12 % fx], 2) +
                   x[13 % fx]*sinf(x[14 % fx]);
            
        case 1: 
            return tanhf(x[0 % fx]+x[1 % fx])*sinf(x[2 % fx]*2 + t) + 
                   logf(fabsf(x[3 % fx])+1)*cosf(x[4 % fx]) + 
                   0.3f*powf(x[5 % fx]-x[6 % fx],3) * expf(-0.15f * t) +
                   expf(-powf(x[7 % fx],2)) +
                   sinf(x[8 % fx]*x[9 % fx]*0.5f) +
                   0.4f*cosf(x[10 % fx] + x[11 % fx] * t) +
                   powf(x[12 % fx]*x[13 % fx], 2) +
                   0.1f*x[14 % fx];
            
        case 2: 
            return expf(-powf(x[0 % fx]-0.5f,2))*sinf(x[1 % fx]*3) + 
                   powf(cosf(x[2 % fx]),2)*x[3 % fx] * (1.0f - expf(-0.1f * t)) + 
                   0.2f*sinhf(x[4 % fx]*x[5 % fx]) +
                   0.5f*tanhf(x[6 % fx] + x[7 % fx]) +
                   powf(x[8 % fx], 3)*0.1f * expf(-0.05f * t) +
                   cosf(x[9 % fx]*x[10 % fx]*(float)M_PI) +
                   0.3f*expf(-powf(x[11 % fx]-x[12 % fx],2)) +
                   0.2f*(x[13 % fx] + x[14 % fx]);
            
        case 3:
            return powf(sinf(x[0 % fx]*x[1 % fx]), 2) +
                   0.4f*tanhf(x[2 % fx] + x[3 % fx]*x[4 % fx] * t) +
                   expf(-fabsf(x[5 % fx]-x[6 % fx])) +
                   0.3f*cosf(x[7 % fx]*x[8 % fx]*2) * expf(-0.2f * t) +
                   powf(x[9 % fx], 2)*sinf(x[10 % fx]) +
                   0.2f*logf(fabsf(x[11 % fx]*x[12 % fx])+1) +
                   0.1f*(x[13 % fx] - x[14 % fx]);
            
        default: 
            return 0.0f;
    }
}

void generate_synthetic_data(float** X, float** y, int num_samples, int input_dim, int output_dim) {
    // Allocate memory
    *X = (float*)malloc(num_samples * input_dim * sizeof(float));
    *y = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Generate random input data
    for (int i = 0; i < num_samples * input_dim; i++) {
        float rand_val = (float)rand() / (float)RAND_MAX;
        (*X)[i] = INPUT_RANGE_MIN + rand_val * (INPUT_RANGE_MAX - INPUT_RANGE_MIN);
    }
    
    // Generate output data using synth_fn
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < output_dim; j++) {
            (*y)[i * output_dim + j] = synth_fn(&(*X)[i * input_dim], i, input_dim, j);
        }
    }
}

void save_data_to_csv(float* X, float* y, int num_samples, int input_dim, int output_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    for (int i = 0; i < input_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    for (int i = 0; i < output_dim - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d\n", output_dim - 1);
    
    // Write data
    for (int i = 0; i < num_samples; i++) {
        // Input features
        for (int j = 0; j < input_dim; j++) {
            fprintf(file, "%.17f,", X[i * input_dim + j]);
        }
        // Output values
        for (int j = 0; j < output_dim - 1; j++) {
            fprintf(file, "%.17f,", y[i * output_dim + j]);
        }
        fprintf(file, "%.17f\n", y[i * output_dim + output_dim - 1]);
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}

void load_csv(const char* filename, float** X, float** y, int* num_samples, int size_x, int size_y) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    
    // Skip header
    char buffer[4096];
    fgets(buffer, sizeof(buffer), file);
    
    // Count lines
    int count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        count++;
    }
    *num_samples = count;
    
    // Allocate memory
    *X = (float*)malloc(count * size_x * sizeof(float));
    *y = (float*)malloc(count * size_y * sizeof(float));
    
    // Reset file pointer and skip header again
    fseek(file, 0, SEEK_SET);
    fgets(buffer, sizeof(buffer), file);
    
    // Read data
    int idx = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        char* token = strtok(buffer, ",");
        for (int i = 0; i < size_x; i++) {
            (*X)[idx * size_x + i] = atof(token);
            token = strtok(NULL, ",");
        }
        for (int i = 0; i < size_y; i++) {
            (*y)[idx * size_y + i] = atof(token);
            token = strtok(NULL, ",");
        }
        idx++;
    }
    
    fclose(file);
}

#endif