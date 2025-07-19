#include "data.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Synthetic function for generating complex outputs (kept for backward compatibility)
float synth_fn(const float* x, int fx, int dim) {
    switch(dim % MAX_SYNTHETIC_OUTPUTS) {
        case 0: 
            return sinf(x[0 % fx]*2)*cosf(x[1 % fx]*1.5f) + 
                   powf(x[2 % fx],2)*x[3 % fx] + 
                   expf(-powf(x[4 % fx]-x[5 % fx],2)) + 
                   0.5f*sinf(x[6 % fx]*x[7 % fx]*(float)M_PI) +
                   tanhf(x[8 % fx] + x[9 % fx]) +
                   0.3f*cosf(x[10 % fx]*x[11 % fx]) +
                   0.2f*powf(x[12 % fx], 2) +
                   x[13 % fx]*sinf(x[14 % fx]);
            
        case 1: 
            return tanhf(x[0 % fx]+x[1 % fx])*sinf(x[2 % fx]*2) + 
                   logf(fabsf(x[3 % fx])+1)*cosf(x[4 % fx]) + 
                   0.3f*powf(x[5 % fx]-x[6 % fx],3) +
                   expf(-powf(x[7 % fx],2)) +
                   sinf(x[8 % fx]*x[9 % fx]*0.5f) +
                   0.4f*cosf(x[10 % fx] + x[11 % fx]) +
                   powf(x[12 % fx]*x[13 % fx], 2) +
                   0.1f*x[14 % fx];
            
        case 2: 
            return expf(-powf(x[0 % fx]-0.5f,2))*sinf(x[1 % fx]*3) + 
                   powf(cosf(x[2 % fx]),2)*x[3 % fx] + 
                   0.2f*sinhf(x[4 % fx]*x[5 % fx]) +
                   0.5f*tanhf(x[6 % fx] + x[7 % fx]) +
                   powf(x[8 % fx], 3)*0.1f +
                   cosf(x[9 % fx]*x[10 % fx]*(float)M_PI) +
                   0.3f*expf(-powf(x[11 % fx]-x[12 % fx],2)) +
                   0.2f*(x[13 % fx] + x[14 % fx]);
            
        case 3:
            return powf(sinf(x[0 % fx]*x[1 % fx]), 2) +
                   0.4f*tanhf(x[2 % fx] + x[3 % fx]*x[4 % fx]) +
                   expf(-fabsf(x[5 % fx]-x[6 % fx])) +
                   0.3f*cosf(x[7 % fx]*x[8 % fx]*2) +
                   powf(x[9 % fx], 2)*sinf(x[10 % fx]) +
                   0.2f*logf(fabsf(x[11 % fx]*x[12 % fx])+1) +
                   0.1f*(x[13 % fx] - x[14 % fx]);
            
        default: 
            return 0.0f;
    }
}

// Generate linear sequence data more suitable for linear state space models
void generate_linear_sequence_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim) {
    // Allocate memory for sequences
    *X = (float*)malloc(num_sequences * seq_len * input_dim * sizeof(float));
    *y = (float*)malloc(num_sequences * seq_len * output_dim * sizeof(float));
    
    // Define linear transformation matrices for more realistic linear dynamics
    float* linear_weights = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* temporal_weights = (float*)malloc(output_dim * sizeof(float));
    
    // Initialize weights for linear combination
    for (int i = 0; i < output_dim * input_dim; i++) {
        linear_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.5f;
    }
    
    // Initialize temporal coefficients
    for (int i = 0; i < output_dim; i++) {
        temporal_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.2f;
    }
    
    for (int seq = 0; seq < num_sequences; seq++) {
        // Generate a coherent sequence with mild temporal dependencies
        float* seq_state = (float*)malloc(input_dim * sizeof(float));
        float* prev_output = (float*)malloc(output_dim * sizeof(float));
        
        // Initialize sequence state
        for (int i = 0; i < input_dim; i++) {
            float rand_val = (float)rand() / (float)RAND_MAX;
            seq_state[i] = INPUT_RANGE_MIN + rand_val * (INPUT_RANGE_MAX - INPUT_RANGE_MIN);
        }
        
        // Initialize previous output
        for (int i = 0; i < output_dim; i++) {
            prev_output[i] = 0.0f;
        }
        
        for (int t = 0; t < seq_len; t++) {
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
            // Smoother temporal evolution for inputs
            for (int i = 0; i < input_dim; i++) {
                // Much more gradual change - more suitable for linear modeling
                seq_state[i] = 0.98f * seq_state[i] + 0.02f * ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
                
                // Clip to input range
                if (seq_state[i] > INPUT_RANGE_MAX) seq_state[i] = INPUT_RANGE_MAX;
                if (seq_state[i] < INPUT_RANGE_MIN) seq_state[i] = INPUT_RANGE_MIN;
                
                (*X)[x_idx + i] = seq_state[i];
            }
            
            // Generate linear outputs with weak temporal dependencies
            for (int j = 0; j < output_dim; j++) {
                float output = 0.0f;
                
                // Linear combination of current inputs
                for (int i = 0; i < input_dim; i++) {
                    output += linear_weights[j * input_dim + i] * (*X)[x_idx + i];
                }
                
                // Add weak temporal component (autoregressive)
                output += 0.1f * prev_output[j];
                
                // Add very weak sinusoidal trend
                float temporal_component = temporal_weights[j] * sinf(2.0f * M_PI * t / (seq_len * 2.0f));
                output += temporal_component;
                
                // Add small amount of noise
                output += ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.05f;
                
                (*y)[y_idx + j] = output;
                prev_output[j] = output;
            }
        }
        
        free(seq_state);
        free(prev_output);
    }
    
    free(linear_weights);
    free(temporal_weights);
}

// Generate complex sequence data (original function)
void generate_synthetic_sequence_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim) {
    // Allocate memory for sequences
    *X = (float*)malloc(num_sequences * seq_len * input_dim * sizeof(float));
    *y = (float*)malloc(num_sequences * seq_len * output_dim * sizeof(float));
    
    for (int seq = 0; seq < num_sequences; seq++) {
        // Generate a coherent sequence with temporal dependencies
        float* seq_state = (float*)malloc(input_dim * sizeof(float));
        
        // Initialize sequence state
        for (int i = 0; i < input_dim; i++) {
            float rand_val = (float)rand() / (float)RAND_MAX;
            seq_state[i] = INPUT_RANGE_MIN + rand_val * (INPUT_RANGE_MAX - INPUT_RANGE_MIN);
        }
        
        for (int t = 0; t < seq_len; t++) {
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
            // More gradual evolution of sequence state
            for (int i = 0; i < input_dim; i++) {
                // Smoother temporal evolution
                seq_state[i] = 0.95f * seq_state[i] + 0.05f * ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
                
                // Clip to input range
                if (seq_state[i] > INPUT_RANGE_MAX) seq_state[i] = INPUT_RANGE_MAX;
                if (seq_state[i] < INPUT_RANGE_MIN) seq_state[i] = INPUT_RANGE_MIN;
                
                (*X)[x_idx + i] = seq_state[i];
            }
            
            // Generate outputs based on current input and time
            for (int j = 0; j < output_dim; j++) {
                float base_output = synth_fn(&(*X)[x_idx], input_dim, j);
                
                // Add temporal component based on position in sequence
                float temporal_component = 0.1f * sinf(2.0f * M_PI * t / seq_len) * (j + 1);
                
                (*y)[y_idx + j] = base_output + temporal_component;
            }
        }
        
        free(seq_state);
    }
}

// Save sequence data to CSV
void save_synthetic_sequence_data_to_csv(float* X, float* y, int num_sequences, int seq_len, int input_dim, int output_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(file, "seq_id,time_step,");
    for (int i = 0; i < input_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    for (int i = 0; i < output_dim - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d\n", output_dim - 1);
    
    // Write data
    for (int seq = 0; seq < num_sequences; seq++) {
        for (int t = 0; t < seq_len; t++) {
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
            fprintf(file, "%d,%d,", seq, t);
            
            // Input features
            for (int j = 0; j < input_dim; j++) {
                fprintf(file, "%.17f,", X[x_idx + j]);
            }
            // Output values
            for (int j = 0; j < output_dim - 1; j++) {
                fprintf(file, "%.17f,", y[y_idx + j]);
            }
            fprintf(file, "%.17f\n", y[y_idx + output_dim - 1]);
        }
    }
    
    fclose(file);
    printf("Sequence data saved to %s\n", filename);
}

// Load CSV sequence data
void load_sequence_csv(const char* filename, float** X, float** y, int* num_sequences, int* seq_len, int input_dim, int output_dim) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    
    // Skip header
    char buffer[4096];
    fgets(buffer, sizeof(buffer), file);
    
    // Count lines and determine dimensions
    int max_seq_id = -1;
    int max_time_step = -1;
    
    while (fgets(buffer, sizeof(buffer), file)) {
        int seq_id, time_step;
        sscanf(buffer, "%d,%d,", &seq_id, &time_step);
        if (seq_id > max_seq_id) max_seq_id = seq_id;
        if (time_step > max_time_step) max_time_step = time_step;
    }
    
    *num_sequences = max_seq_id + 1;
    *seq_len = max_time_step + 1;
    
    // Allocate memory
    *X = (float*)malloc((*num_sequences) * (*seq_len) * input_dim * sizeof(float));
    *y = (float*)malloc((*num_sequences) * (*seq_len) * output_dim * sizeof(float));
    
    // Reset file pointer and skip header again
    fseek(file, 0, SEEK_SET);
    fgets(buffer, sizeof(buffer), file);
    
    // Read data
    while (fgets(buffer, sizeof(buffer), file)) {
        int seq_id, time_step;
        char* ptr = buffer;
        
        // Parse seq_id and time_step
        seq_id = strtol(ptr, &ptr, 10);
        ptr++; // skip comma
        time_step = strtol(ptr, &ptr, 10);
        ptr++; // skip comma
        
        int x_idx = seq_id * (*seq_len) * input_dim + time_step * input_dim;
        int y_idx = seq_id * (*seq_len) * output_dim + time_step * output_dim;
        
        // Parse input features
        for (int i = 0; i < input_dim; i++) {
            (*X)[x_idx + i] = strtof(ptr, &ptr);
            ptr++; // skip comma
        }
        
        // Parse output values
        for (int i = 0; i < output_dim; i++) {
            (*y)[y_idx + i] = strtof(ptr, &ptr);
            if (i < output_dim - 1) ptr++; // skip comma
        }
    }
    
    fclose(file);
}

// Reshape data from [batch][time][feature] to [time][batch][feature]
void reshape_data_for_batch_processing(float* X, float* y, 
                                     float** X_reshaped, float** y_reshaped,
                                     int num_sequences, int seq_len, 
                                     int input_dim, int output_dim) {
    // Reshape to: seq_len tensors of size (batch_size x input_dim/output_dim)
    *X_reshaped = (float*)malloc(seq_len * num_sequences * input_dim * sizeof(float));
    *y_reshaped = (float*)malloc(seq_len * num_sequences * output_dim * sizeof(float));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            // Original layout: [seq][time][feature]
            int orig_x_idx = b * seq_len * input_dim + t * input_dim;
            int orig_y_idx = b * seq_len * output_dim + t * output_dim;
            
            // New layout: [time][seq][feature] 
            int new_x_idx = t * num_sequences * input_dim + b * input_dim;
            int new_y_idx = t * num_sequences * output_dim + b * output_dim;
            
            memcpy(&(*X_reshaped)[new_x_idx], &X[orig_x_idx], input_dim * sizeof(float));
            memcpy(&(*y_reshaped)[new_y_idx], &y[orig_y_idx], output_dim * sizeof(float));
        }
    }
}