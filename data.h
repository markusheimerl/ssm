#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SYNTHETIC_OUTPUTS 4
#define INPUT_RANGE_MIN -3.0f
#define INPUT_RANGE_MAX 3.0f
#define OUTPUT_RANGE_MIN -20.0f
#define OUTPUT_RANGE_MAX 20.0f

// Helper function to scale output to desired range
float scale_output(float x) {
    // First squash with tanh to get [-1, 1]
    float squashed = tanhf(x);
    // Then scale to [OUTPUT_RANGE_MIN, OUTPUT_RANGE_MAX]
    return OUTPUT_RANGE_MIN + (squashed + 1.0f) * 0.5f * 
           (OUTPUT_RANGE_MAX - OUTPUT_RANGE_MIN);
}

float temporal_pattern(int t, int pattern_type, float freq) {
    switch(pattern_type % 4) {
        case 0: // Damped oscillation
            return sinf(freq * t * 0.1f) * expf(-0.05f * t);
        case 1: // Growing oscillation
            return sinf(freq * t * 0.1f) * (1.0f - expf(-0.05f * t));
        case 2: // Periodic with phase shift
            return sinf(freq * t * 0.1f + cosf(t * 0.05f));
        case 3: // Chaotic-like pattern
            return sinf(freq * t * 0.1f + sinf(t * 0.03f));
        default:
            return 0.0f;
    }
}

float synth_fn(const float* x, int seq_idx, int fx, int dim) {
    float seq_t = seq_idx * 0.1f;
    float temporal_base = temporal_pattern(seq_idx, dim, 1.0f);
    
    float raw_output;
    switch(dim % MAX_SYNTHETIC_OUTPUTS) {
        case 0: 
            raw_output = temporal_base * (
                sinf(x[0 % fx]*2.0f + seq_t)*cosf(x[1 % fx]*1.5f) + 
                expf(-powf(x[2 % fx]-x[3 % fx], 2))
            );
            break;
            
        case 1:
            raw_output = temporal_base * (
                tanhf(x[0 % fx] + x[1 % fx]) + 
                logf(fabsf(x[2 % fx])+1.0f)*cosf(x[3 % fx]*seq_t)
            );
            break;
            
        case 2:
            raw_output = temporal_base * (
                expf(-powf(x[0 % fx]-0.5f, 2)) + 
                0.2f*sinhf(x[1 % fx]*x[2 % fx]*seq_t)
            );
            break;
            
        case 3:
            raw_output = temporal_base * (
                powf(sinf(x[0 % fx]*x[1 % fx]*seq_t), 2) +
                0.3f*cosf(x[2 % fx]*x[3 % fx]*seq_t)
            );
            break;
            
        default: 
            raw_output = 0.0f;
    }
    
    return scale_output(raw_output);
}

void generate_synthetic_data(float** X, float** y, int num_samples, int input_dim, int output_dim, int seq_len) {
    // Allocate memory
    *X = (float*)malloc(num_samples * input_dim * sizeof(float));
    *y = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Generate random input data with some temporal smoothing
    float* smooth_factors = (float*)malloc(input_dim * sizeof(float));
    float* prev_values = (float*)malloc(input_dim * sizeof(float));
    
    // Initialize smoothing factors for each input dimension
    for (int i = 0; i < input_dim; i++) {
        smooth_factors[i] = 0.1f + 0.8f * ((float)rand() / (float)RAND_MAX);
        prev_values[i] = INPUT_RANGE_MIN + 
                        (INPUT_RANGE_MAX - INPUT_RANGE_MIN) * ((float)rand() / (float)RAND_MAX);
    }
    
    // Generate temporally smooth input data
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_dim; j++) {
            float target = INPUT_RANGE_MIN + 
                          (INPUT_RANGE_MAX - INPUT_RANGE_MIN) * ((float)rand() / (float)RAND_MAX);
            float smooth_val = prev_values[j] * (1.0f - smooth_factors[j]) + 
                             target * smooth_factors[j];
            (*X)[i * input_dim + j] = smooth_val;
            prev_values[j] = smooth_val;
        }
    }
    
    // Generate output data using synth_fn with sequence dependencies
    for (int i = 0; i < num_samples; i++) {
        int seq_idx = i % seq_len;  // Position within the sequence
        
        for (int j = 0; j < output_dim; j++) {
            (*y)[i * output_dim + j] = synth_fn(&(*X)[i * input_dim], 
                                               seq_idx, input_dim, j);
        }
    }
    
    free(smooth_factors);
    free(prev_values);
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