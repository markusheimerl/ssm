#include "data.h"

static float evaluate_synthetic_function(int num_terms, const float* coefficients, const int* operations,
                              const int* idx1, const int* idx2, const int* add_subtract, const float* x) {
    float result = 0.0f;
    
    for (int i = 0; i < num_terms; i++) {
        float coefficient = coefficients[i];
        int operation = operations[i];
        int input_idx1 = idx1[i];
        int input_idx2 = idx2[i];
        int add_sub = add_subtract[i];
        
        float term_value = 0.0f;
        
        switch (operation) {
            case 0: term_value = coefficient * sinf(x[input_idx1] * 2.0f); break;
            case 1: term_value = coefficient * cosf(x[input_idx1] * 1.5f); break;
            case 2: term_value = coefficient * tanhf(x[input_idx1] + x[input_idx2]); break;
            case 3: term_value = coefficient * expf(-powf(x[input_idx1], 2)); break;
            case 4: term_value = coefficient * logf(fabsf(x[input_idx1]) + 1.0f); break;
            case 5: term_value = coefficient * powf(x[input_idx1], 2) * x[input_idx2]; break;
            case 6: term_value = coefficient * sinhf(x[input_idx1] * x[input_idx2]); break;
            case 7: term_value = coefficient * x[input_idx1] * sinf(x[input_idx2] * M_PI); break;
        }
        
        if (add_sub == 0) {
            result += term_value;
        } else {
            result -= term_value;
        }
    }
    
    return result;
}

static float evaluate_temporal_function(int num_terms, const float* coefficients, const int* operations,
                                       const int* idx1, const int* idx2, const int* add_subtract, 
                                       float normalized_time, const float* current_input, 
                                       const float* prev_output, int input_dim, int output_dim) {
    float result = 0.0f;
    
    for (int i = 0; i < num_terms; i++) {
        float coefficient = coefficients[i];
        int operation = operations[i];
        int input_idx1 = idx1[i] % input_dim;  // Clamp to input dimensions
        int input_idx2 = idx2[i] % input_dim;
        int output_idx1 = idx1[i] % output_dim; // For accessing previous outputs
        int add_sub = add_subtract[i];
        
        float term_value = 0.0f;
        
        switch (operation) {
            case 0: // Temporal sine wave
                term_value = coefficient * sinf(normalized_time * 2.0f * M_PI + current_input[input_idx1]); 
                break;
            case 1: // Temporal cosine with input modulation
                term_value = coefficient * cosf(normalized_time * M_PI + current_input[input_idx1] * 0.5f); 
                break;
            case 2: // Previous output dependency with time
                term_value = coefficient * prev_output[output_idx1] * sinf(normalized_time * 4.0f * M_PI); 
                break;
            case 3: // Exponential temporal decay with input
                term_value = coefficient * expf(-normalized_time * 2.0f) * current_input[input_idx1]; 
                break;
            case 4: // Temporal growth with saturation
                term_value = coefficient * tanhf(normalized_time * 3.0f + current_input[input_idx1]); 
                break;
            case 5: // Cross-temporal interaction
                term_value = coefficient * current_input[input_idx1] * prev_output[output_idx1] * cosf(normalized_time * M_PI); 
                break;
            case 6: // Phase-shifted temporal wave
                term_value = coefficient * sinf(normalized_time * M_PI + current_input[input_idx1] + current_input[input_idx2]); 
                break;
            case 7: // Memory-like temporal function
                term_value = coefficient * prev_output[output_idx1] * expf(-powf(normalized_time - 0.5f, 2) * 2.0f); 
                break;
        }
        
        if (add_sub == 0) {
            result += term_value;
        } else {
            result -= term_value;
        }
    }
    
    return result;
}

void generate_synthetic_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim, 
                           float input_min, float input_max) {
    // Allocate memory for sequences
    *X = (float*)malloc(num_sequences * seq_len * input_dim * sizeof(float));
    *y = (float*)malloc(num_sequences * seq_len * output_dim * sizeof(float));
    
    // Create base function parameters for each output dimension
    int* num_terms_per_output = (int*)malloc(output_dim * sizeof(int));
    float** coefficients = (float**)malloc(output_dim * sizeof(float*));
    int** operations = (int**)malloc(output_dim * sizeof(int*));
    int** idx1 = (int**)malloc(output_dim * sizeof(int*));
    int** idx2 = (int**)malloc(output_dim * sizeof(int*));
    int** add_subtract = (int**)malloc(output_dim * sizeof(int*));
    
    // Create temporal function parameters for each output dimension
    int* num_temporal_terms_per_output = (int*)malloc(output_dim * sizeof(int));
    float** temporal_coefficients = (float**)malloc(output_dim * sizeof(float*));
    int** temporal_operations = (int**)malloc(output_dim * sizeof(int*));
    int** temporal_idx1 = (int**)malloc(output_dim * sizeof(int*));
    int** temporal_idx2 = (int**)malloc(output_dim * sizeof(int*));
    int** temporal_add_subtract = (int**)malloc(output_dim * sizeof(int*));
    
    for (int output_idx = 0; output_idx < output_dim; output_idx++) {
        // Base function parameters
        int num_terms = 6 + (rand() % 7);
        num_terms_per_output[output_idx] = num_terms;
        
        coefficients[output_idx] = (float*)malloc(num_terms * sizeof(float));
        operations[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx1[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx2[output_idx] = (int*)malloc(num_terms * sizeof(int));
        add_subtract[output_idx] = (int*)malloc(num_terms * sizeof(int));

        for (int term = 0; term < num_terms; term++) {
            coefficients[output_idx][term] = 0.1f + 0.4f * ((float)rand() / (float)RAND_MAX);
            operations[output_idx][term] = rand() % 8;
            idx1[output_idx][term] = rand() % input_dim;
            idx2[output_idx][term] = rand() % input_dim;
            add_subtract[output_idx][term] = rand() % 2;
        }
        
        // Temporal function parameters
        int num_temporal_terms = 3 + (rand() % 5); // 3-7 temporal terms
        num_temporal_terms_per_output[output_idx] = num_temporal_terms;
        
        temporal_coefficients[output_idx] = (float*)malloc(num_temporal_terms * sizeof(float));
        temporal_operations[output_idx] = (int*)malloc(num_temporal_terms * sizeof(int));
        temporal_idx1[output_idx] = (int*)malloc(num_temporal_terms * sizeof(int));
        temporal_idx2[output_idx] = (int*)malloc(num_temporal_terms * sizeof(int));
        temporal_add_subtract[output_idx] = (int*)malloc(num_temporal_terms * sizeof(int));

        for (int term = 0; term < num_temporal_terms; term++) {
            temporal_coefficients[output_idx][term] = 0.05f + 0.15f * ((float)rand() / (float)RAND_MAX); // Smaller coefficients for temporal
            temporal_operations[output_idx][term] = rand() % 8;
            temporal_idx1[output_idx][term] = rand() % (input_dim + output_dim); // Can reference inputs or outputs
            temporal_idx2[output_idx][term] = rand() % (input_dim + output_dim);
            temporal_add_subtract[output_idx][term] = rand() % 2;
        }
    }
    
    for (int seq = 0; seq < num_sequences; seq++) {
        // Generate a coherent sequence with temporal dependencies
        float* seq_state = (float*)malloc(input_dim * sizeof(float));
        float* prev_output = (float*)calloc(output_dim, sizeof(float)); // Initialize to zero
        
        // Initialize sequence state
        for (int i = 0; i < input_dim; i++) {
            float rand_val = (float)rand() / (float)RAND_MAX;
            seq_state[i] = input_min + rand_val * (input_max - input_min);
        }
        
        for (int t = 0; t < seq_len; t++) {
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
            // Evolve sequence state with temporal smoothness
            for (int i = 0; i < input_dim; i++) {
                // Smoother temporal evolution with some randomness
                float evolution_factor = 0.95f + 0.05f * sinf(2.0f * M_PI * t / seq_len); // Slightly varying smoothness
                seq_state[i] = evolution_factor * seq_state[i] + (1.0f - evolution_factor) * ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
                
                // Clip to input range
                if (seq_state[i] > input_max) seq_state[i] = input_max;
                if (seq_state[i] < input_min) seq_state[i] = input_min;
                
                (*X)[x_idx + i] = seq_state[i];
            }
            
            // Generate outputs using both base and temporal functions
            for (int j = 0; j < output_dim; j++) {
                // Base output from spatial function
                float base_output = evaluate_synthetic_function(num_terms_per_output[j], 
                                                             coefficients[j], operations[j], 
                                                             idx1[j], idx2[j], add_subtract[j], 
                                                             &(*X)[x_idx]);
                
                // Temporal component using sophisticated temporal function
                float normalized_time = (float)t / (seq_len - 1); // 0 to 1
                float temporal_component = evaluate_temporal_function(num_temporal_terms_per_output[j],
                                                                    temporal_coefficients[j], temporal_operations[j],
                                                                    temporal_idx1[j], temporal_idx2[j], temporal_add_subtract[j],
                                                                    normalized_time, &(*X)[x_idx], prev_output,
                                                                    input_dim, output_dim);
                
                (*y)[y_idx + j] = base_output + temporal_component;
                
                // Update previous output for next time step
                prev_output[j] = (*y)[y_idx + j];
            }
        }
        
        free(seq_state);
        free(prev_output);
    }
    
    // Clean up base function parameters
    for (int i = 0; i < output_dim; i++) {
        free(coefficients[i]);
        free(operations[i]);
        free(idx1[i]);
        free(idx2[i]);
        free(add_subtract[i]);
    }
    free(num_terms_per_output);
    free(coefficients);
    free(operations);
    free(idx1);
    free(idx2);
    free(add_subtract);
    
    // Clean up temporal function parameters
    for (int i = 0; i < output_dim; i++) {
        free(temporal_coefficients[i]);
        free(temporal_operations[i]);
        free(temporal_idx1[i]);
        free(temporal_idx2[i]);
        free(temporal_add_subtract[i]);
    }
    free(num_temporal_terms_per_output);
    free(temporal_coefficients);
    free(temporal_operations);
    free(temporal_idx1);
    free(temporal_idx2);
    free(temporal_add_subtract);
}

void save_data(float* X, float* y, int num_sequences, int seq_len, int input_dim, int output_dim, const char* filename) {
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
    printf("Data saved to %s\n", filename);
}

void load_data(const char* filename, float** X, float** y, int* num_sequences, int* seq_len, int input_dim, int output_dim) {
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
