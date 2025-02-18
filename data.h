#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SYNTHETIC_OUTPUTS 4
#define HIDDEN_STATE_DIM 8
#define INPUT_RANGE_MIN -3.0f
#define INPUT_RANGE_MAX 3.0f
#define OUTPUT_RANGE_MIN -20.0f
#define OUTPUT_RANGE_MAX 20.0f

// Hidden state for synthetic data generation
typedef struct {
    float values[HIDDEN_STATE_DIM];  // Internal state vector
} HiddenState;

// Helper function to scale output to desired range
float scale_output(float x) {
    float squashed = tanhf(x);
    return OUTPUT_RANGE_MIN + (squashed + 1.0f) * 0.5f * 
           (OUTPUT_RANGE_MAX - OUTPUT_RANGE_MIN);
}

// Update hidden state based on input
void update_hidden_state(HiddenState* state, const float* x, int input_dim) {
    // Each input contributes to updating the hidden state
    for(int i = 0; i < HIDDEN_STATE_DIM; i++) {
        float sum = 0.0f;
        for(int j = 0; j < input_dim; j++) {
            // Different nonlinear contributions from each input
            sum += sinf(state->values[i] + x[j]) * 0.1f;
            sum += cosf(x[j] * state->values[(i+1)%HIDDEN_STATE_DIM]) * 0.1f;
        }
        // Update state with some decay
        state->values[i] = state->values[i] * 0.95f + sum;
    }
}

// Generate outputs based on current hidden state and inputs
void generate_outputs(const HiddenState* state, const float* x, int input_dim, float* y, int output_dim) {
    for(int i = 0; i < output_dim; i++) {
        float state_sum = 0.0f;
        float input_sum = 0.0f;

        // Contribution from hidden state
        switch(i) {
            case 0:
                state_sum = sinf(state->values[0]) * state->values[1] + 
                           cosf(state->values[2] * state->values[3]);
                break;
            case 1:
                state_sum = state->values[4 % HIDDEN_STATE_DIM] * 
                           sinf(state->values[5 % HIDDEN_STATE_DIM]) + 
                           state->values[6 % HIDDEN_STATE_DIM] * 
                           state->values[7 % HIDDEN_STATE_DIM];
                break;
            case 2:
                state_sum = sinf(state->values[0] * state->values[4 % HIDDEN_STATE_DIM]) * 
                           cosf(state->values[2] * state->values[6 % HIDDEN_STATE_DIM]);
                break;
            case 3:
                state_sum = state->values[1] * state->values[3] * 
                           sinf(state->values[5 % HIDDEN_STATE_DIM] * 
                           state->values[7 % HIDDEN_STATE_DIM]);
                break;
        }

        // Contribution from current inputs
        for(int j = 0; j < input_dim; j++) {
            switch(i) {
                case 0:
                    input_sum += sinf(x[j]) * cosf(x[(j+1) % input_dim]);
                    break;
                case 1:
                    input_sum += x[j] * sinf(x[(j+2) % input_dim]);
                    break;
                case 2:
                    input_sum += cosf(x[j] * x[(j+1) % input_dim]);
                    break;
                case 3:
                    input_sum += sinf(x[j] + x[(j+2) % input_dim]);
                    break;
            }
        }
        
        // Combine state and input contributions
        float combined = 0.7f * state_sum + 0.3f * input_sum;
        y[i] = scale_output(combined);
    }
}

void generate_synthetic_data(float** X, float** y, int num_samples, int input_dim, 
                           int output_dim, int seq_len) {
    // Allocate memory
    *X = (float*)malloc(num_samples * input_dim * sizeof(float));
    *y = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Initialize hidden state
    HiddenState state;
    memset(&state, 0, sizeof(HiddenState));
    
    // Generate data
    for(int i = 0; i < num_samples; i++) {
        // Generate random inputs
        for(int j = 0; j < input_dim; j++) {
            (*X)[i * input_dim + j] = INPUT_RANGE_MIN + 
                (INPUT_RANGE_MAX - INPUT_RANGE_MIN) * 
                ((float)rand() / (float)RAND_MAX);
        }
        
        // Reset state at sequence boundaries
        if(i % seq_len == 0) {
            memset(&state, 0, sizeof(HiddenState));
        }
        
        // Update hidden state with current input
        update_hidden_state(&state, &(*X)[i * input_dim], input_dim);
        
        // Generate outputs based on current state and inputs
        generate_outputs(&state, &(*X)[i * input_dim], input_dim, 
                        &(*y)[i * output_dim], output_dim);
    }
}

void save_data_to_csv(float* X, float* y, int num_samples, int input_dim, 
                     int output_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    for(int i = 0; i < input_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    for(int i = 0; i < output_dim - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d\n", output_dim - 1);
    
    // Write data
    for(int i = 0; i < num_samples; i++) {
        for(int j = 0; j < input_dim; j++) {
            fprintf(file, "%.17f,", X[i * input_dim + j]);
        }
        for(int j = 0; j < output_dim - 1; j++) {
            fprintf(file, "%.17f,", y[i * output_dim + j]);
        }
        fprintf(file, "%.17f\n", y[i * output_dim + output_dim - 1]);
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}

void load_csv(const char* filename, float** X, float** y, int* num_samples, 
             int size_x, int size_y) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    
    char buffer[4096];
    fgets(buffer, sizeof(buffer), file);
    
    int count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        count++;
    }
    *num_samples = count;
    
    *X = (float*)malloc(count * size_x * sizeof(float));
    *y = (float*)malloc(count * size_y * sizeof(float));
    
    fseek(file, 0, SEEK_SET);
    fgets(buffer, sizeof(buffer), file);
    
    int idx = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        char* token = strtok(buffer, ",");
        for(int i = 0; i < size_x; i++) {
            (*X)[idx * size_x + i] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int i = 0; i < size_y; i++) {
            (*y)[idx * size_y + i] = atof(token);
            token = strtok(NULL, ",");
        }
        idx++;
    }
    
    fclose(file);
}

#endif