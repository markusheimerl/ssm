#include "data.h"

static float evaluate_synthetic_function(int num_terms, const float* coefficients, const int* operations,
                              const int* idx1, const int* idx2, const int* add_subtract, 
                              const int* time_lags, const float* X, int seq, int t, int seq_len, int input_dim) {
    float result = 0.0f;
    
    for (int i = 0; i < num_terms; i++) {
        float coefficient = coefficients[i];
        int operation = operations[i];
        int input_idx1 = idx1[i];
        int input_idx2 = idx2[i];
        int add_sub = add_subtract[i];
        int lag = time_lags[i];
        
        // Skip if we don't have enough history
        if (t < lag) continue;
        
        // Get input values from t-lag timestep
        int x_base_idx = seq * seq_len * input_dim + (t - lag) * input_dim;
        const float* x = &X[x_base_idx];
        
        float term_value = 0.0f;
        
        switch (operation) {
            case 0: term_value = coefficient * sinf(x[input_idx1] * 2.0f); break;
            case 1: term_value = coefficient * cosf(x[input_idx1] * 1.5f); break;
            case 2: term_value = coefficient * tanhf(x[input_idx1] + x[input_idx2]); break;
            case 3: term_value = coefficient * expf(-powf(x[input_idx1], 2)); break;
            case 4: term_value = coefficient * logf(fabsf(x[input_idx1]) + 1.0f); break;
            case 5: term_value = coefficient * powf(x[input_idx1], 2) * x[input_idx2]; break;
            case 6: term_value = coefficient * sinhf(x[input_idx1] - x[input_idx2]); break;
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

static void print_symbolic_function(int output_idx, int num_terms, const float* coefficients, 
                                  const int* operations, const int* idx1, const int* idx2, 
                                  const int* add_subtract, const int* time_lags) {
    printf("y%d = ", output_idx);
    
    for (int i = 0; i < num_terms; i++) {
        float coeff = coefficients[i];
        int op = operations[i];
        int in1 = idx1[i];
        int in2 = idx2[i];
        int add_sub = add_subtract[i];
        int lag = time_lags[i];
        
        // Print sign
        if (i == 0) {
            if (add_sub == 1) printf("-");
        } else {
            printf(" %s ", (add_sub == 0) ? "+" : "-");
        }
        
        // Print coefficient if not 1.0
        if (fabsf(coeff - 1.0f) > 1e-6) {
            printf("%.3f*", coeff);
        }
        
        // Print operation
        switch (op) {
            case 0: printf(lag == 0 ? "sin(2*x%d)" : "sin(2*x%d[t-%d])", in1, lag); break;
            case 1: printf(lag == 0 ? "cos(1.5*x%d)" : "cos(1.5*x%d[t-%d])", in1, lag); break;
            case 2: printf(lag == 0 ? "tanh(x%d + x%d)" : "tanh(x%d[t-%d] + x%d[t-%d])", in1, lag, in2, lag); break;
            case 3: printf(lag == 0 ? "exp(-x%d^2)" : "exp(-x%d[t-%d]^2)", in1, lag); break;
            case 4: printf(lag == 0 ? "log(|x%d| + 1)" : "log(|x%d[t-%d]| + 1)", in1, lag); break;
            case 5: printf(lag == 0 ? "x%d^2*x%d" : "x%d[t-%d]^2*x%d[t-%d]", in1, lag, in2, lag); break;
            case 6: printf(lag == 0 ? "sinh(x%d-x%d)" : "sinh(x%d[t-%d]-x%d[t-%d])", in1, lag, in2, lag); break;
            case 7: printf(lag == 0 ? "x%d*sin(π*x%d)" : "x%d[t-%d]*sin(π*x%d[t-%d])", in1, lag, in2, lag); break;
        }
    }
    printf("\n");
}

void generate_synthetic_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim, 
                           float input_min, float input_max) {
    // Allocate memory
    *X = (float*)malloc(num_sequences * seq_len * input_dim * sizeof(float));
    *y = (float*)malloc(num_sequences * seq_len * output_dim * sizeof(float));
    
    // Generate random input data
    for (int i = 0; i < num_sequences * seq_len * input_dim; i++) {
        float rand_val = (float)rand() / (float)RAND_MAX;
        (*X)[i] = input_min + rand_val * (input_max - input_min);
    }
    
    // Create function parameters for each output dimension
    int* num_terms_per_output = (int*)malloc(output_dim * sizeof(int));
    float** coefficients = (float**)malloc(output_dim * sizeof(float*));
    int** operations = (int**)malloc(output_dim * sizeof(int*));
    int** idx1 = (int**)malloc(output_dim * sizeof(int*));
    int** idx2 = (int**)malloc(output_dim * sizeof(int*));
    int** add_subtract = (int**)malloc(output_dim * sizeof(int*));
    int** time_lags = (int**)malloc(output_dim * sizeof(int*));
    
    for (int output_idx = 0; output_idx < output_dim; output_idx++) {
        // Random number of terms between 2 and 4 (reduced for linear model)
        int num_terms = 2 + (rand() % 3);
        num_terms_per_output[output_idx] = num_terms;
        
        // Allocate arrays for this function's terms
        coefficients[output_idx] = (float*)malloc(num_terms * sizeof(float));
        operations[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx1[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx2[output_idx] = (int*)malloc(num_terms * sizeof(int));
        add_subtract[output_idx] = (int*)malloc(num_terms * sizeof(int));
        time_lags[output_idx] = (int*)malloc(num_terms * sizeof(int));

        // Generate random terms with gentler coefficients
        for (int term = 0; term < num_terms; term++) {
            coefficients[output_idx][term] = 0.05f + 0.15f * ((float)rand() / (float)RAND_MAX); // Smaller coefficients
            operations[output_idx][term] = rand() % 3; // Only use gentler operations (sin, cos, tanh)
            idx1[output_idx][term] = rand() % input_dim;
            idx2[output_idx][term] = rand() % input_dim;
            add_subtract[output_idx][term] = rand() % 2;
            time_lags[output_idx][term] = rand() % 3; // Shorter time lags
        }
    }
    
    // Print symbolic representation of generated functions
    printf("\nGenerated synthetic functions:\n");
    for (int output_idx = 0; output_idx < output_dim; output_idx++) {
        print_symbolic_function(output_idx, num_terms_per_output[output_idx], 
                              coefficients[output_idx], operations[output_idx], 
                              idx1[output_idx], idx2[output_idx], add_subtract[output_idx],
                              time_lags[output_idx]);
    }
    printf("\n");
    
    // Generate output data by evaluating each function
    for (int seq = 0; seq < num_sequences; seq++) {
        for (int t = 0; t < seq_len; t++) {
            for (int j = 0; j < output_dim; j++) {
                int y_idx = seq * seq_len * output_dim + t * output_dim + j;
                (*y)[y_idx] = evaluate_synthetic_function(num_terms_per_output[j], 
                                                        coefficients[j], operations[j], 
                                                        idx1[j], idx2[j], add_subtract[j],
                                                        time_lags[j], *X, seq, t, seq_len, input_dim);
            }
        }
    }
    
    // Clean up
    for (int i = 0; i < output_dim; i++) {
        free(coefficients[i]);
        free(operations[i]);
        free(idx1[i]);
        free(idx2[i]);
        free(add_subtract[i]);
        free(time_lags[i]);
    }
    free(num_terms_per_output);
    free(coefficients);
    free(operations);
    free(idx1);
    free(idx2);
    free(add_subtract);
    free(time_lags);
}

void save_data(float* X, float* y, int num_sequences, int seq_len, int input_dim, int output_dim, const char* filename) {
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
    for (int seq = 0; seq < num_sequences; seq++) {
        for (int t = 0; t < seq_len; t++) {
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
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

        if (seq < num_sequences - 1) {
            fprintf(file, "\n");
        }
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}