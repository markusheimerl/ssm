#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "data.h"
#include "ssm.h"

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

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int input_dim = 16;
    const int state_dim = 128;
    const int output_dim = 4;
    const int seq_len = 32;
    const int num_sequences = 32;
    const int batch_size = num_sequences;
    
    // Generate synthetic sequence data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_sequences, seq_len, input_dim, output_dim, -3.0f, 3.0f);
    
    // Reshape data for batch processing
    float *X_reshaped, *y_reshaped;
    reshape_data_for_batch_processing(X, y, &X_reshaped, &y_reshaped,
                                    num_sequences, seq_len, input_dim, output_dim);
    
    // Initialize state space models for both approaches
    SSM* ssm_sequential = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    SSM* ssm_parallel = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Copy identical weights to both models for fair comparison
    memcpy(ssm_parallel->A, ssm_sequential->A, state_dim * state_dim * sizeof(float));
    memcpy(ssm_parallel->B, ssm_sequential->B, state_dim * input_dim * sizeof(float));
    memcpy(ssm_parallel->C, ssm_sequential->C, output_dim * state_dim * sizeof(float));
    memcpy(ssm_parallel->D, ssm_sequential->D, output_dim * input_dim * sizeof(float));
    
    // Training parameters
    const int num_epochs = 100;  // Reduced for testing
    const float learning_rate = 0.0003f;
    
    printf("\n=== SEQUENTIAL CPU TRAINING ===\n");
    double sequential_start = omp_get_wtime();
    
    // Sequential training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass (sequential)
        reset_state_ssm(ssm_sequential);
        for (int t = 0; t < seq_len; t++) {
            float* X_t = X_reshaped + t * batch_size * input_dim;
            forward_pass_ssm(ssm_sequential, X_t, t);
        }
        
        // Calculate loss
        float loss = calculate_loss_ssm(ssm_sequential, y_reshaped);

        // Print progress
        if (epoch > 0 && epoch % 500 == 0) {
            printf("Sequential Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_ssm(ssm_sequential);
        backward_pass_ssm(ssm_sequential, X_reshaped);
        
        // Update weights
        update_weights_ssm(ssm_sequential, learning_rate);
    }
    
    double sequential_end = omp_get_wtime();
    double sequential_time = sequential_end - sequential_start;
    
    printf("\n=== PARALLEL CPU TRAINING (Blelloch Scan) ===\n");
    double parallel_start = omp_get_wtime();
    
    // Parallel training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass (parallel scan)
        reset_state_ssm(ssm_parallel);
        forward_pass_ssm_parallel(ssm_parallel, X_reshaped);
        
        // Calculate loss
        float loss = calculate_loss_ssm(ssm_parallel, y_reshaped);

        // Print progress
        if (epoch > 0 && epoch % 500 == 0) {
            printf("Parallel Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_ssm(ssm_parallel);
        backward_pass_ssm(ssm_parallel, X_reshaped);
        
        // Update weights
        update_weights_ssm(ssm_parallel, learning_rate);
    }
    
    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;
    
    // Print timing results
    printf("\n=== TIMING COMPARISON ===\n");
    printf("Sequential training time: %.3f seconds\n", sequential_time);
    printf("Parallel training time:   %.3f seconds\n", parallel_time);
    printf("Speedup: %.2fx\n", sequential_time / parallel_time);
    
    // Verify numerical correctness by comparing final predictions
    printf("\n=== NUMERICAL VERIFICATION ===\n");
    
    // Test both models on the same data
    reset_state_ssm(ssm_sequential);
    for (int t = 0; t < seq_len; t++) {
        float* X_t = X_reshaped + t * batch_size * input_dim;
        forward_pass_ssm(ssm_sequential, X_t, t);
    }
    float sequential_loss = calculate_loss_ssm(ssm_sequential, y_reshaped);
    
    reset_state_ssm(ssm_parallel);
    forward_pass_ssm_parallel(ssm_parallel, X_reshaped);
    float parallel_loss = calculate_loss_ssm(ssm_parallel, y_reshaped);
    
    printf("Final sequential loss: %.8f\n", sequential_loss);
    printf("Final parallel loss:   %.8f\n", parallel_loss);
    printf("Loss difference:       %.2e\n", fabs(sequential_loss - parallel_loss));
    
    // Compare a few predictions to ensure they're similar
    printf("\nSample prediction comparison (first 5 outputs of first timestep):\n");
    for (int i = 0; i < 5 && i < ssm_sequential->output_dim; i++) {
        float seq_pred = ssm_sequential->predictions[i];
        float par_pred = ssm_parallel->predictions[i];
        printf("Output %d: Sequential=%.6f, Parallel=%.6f, Diff=%.2e\n", 
               i, seq_pred, par_pred, fabs(seq_pred - par_pred));
    }
    
    // Use the sequential model for final output (as it's the reference)
    SSM* ssm = ssm_sequential;

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&now));

    // Save model and data with timestamped filenames
    save_ssm(ssm, model_fname);
    save_data(X, y, num_sequences, seq_len, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    SSM* loaded_ssm = load_ssm(model_fname, batch_size);
    
    // Forward pass with loaded model
    reset_state_ssm(loaded_ssm);
    for (int t = 0; t < seq_len; t++) {
        float* X_t = X_reshaped + t * batch_size * input_dim;
        forward_pass_ssm(loaded_ssm, X_t, t);
    }
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_ssm(loaded_ssm, y_reshaped);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores
    printf("\nR² scores:\n");
    int total_samples = num_sequences * seq_len;
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                y_mean += y_reshaped[idx];
            }
        }
        y_mean /= total_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                float diff_res = y_reshaped[idx] - loaded_ssm->predictions[idx];
                float diff_tot = y_reshaped[idx] - y_mean;
                ss_res += diff_res * diff_res;
                ss_tot += diff_tot * diff_tot;
            }
        }
        float r2 = 1.0f - (ss_res / ss_tot);
        printf("R² score for output y%d: %.8f\n", i, r2);
    }

    // Print sample predictions from first sequence
    printf("\nSample Predictions (first sequence, first 10 time steps):\n");
    printf("Time\tOutput\t\tPredicted\tActual\t\tDifference\n");
    printf("----------------------------------------------------------------\n");

    for (int i = 0; i < output_dim; i++) {
        printf("\ny%d:\n", i);
        for (int t = 0; t < 10; t++) {
            // First sequence (b=0) in reshaped format
            int idx = t * num_sequences * output_dim + 0 * output_dim + i;
            float pred = loaded_ssm->predictions[idx];
            float actual = y_reshaped[idx];
            float diff = pred - actual;
            printf("t=%d\t\t%8.3f\t%8.3f\t%8.3f\n", t, pred, actual, diff);
        }
        
        // Calculate MAE for this output across all sequences and time steps
        float mae = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                mae += fabs(loaded_ssm->predictions[idx] - y_reshaped[idx]);
            }
        }
        mae /= total_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_reshaped);
    free(y_reshaped);
    free_ssm(ssm);
    free_ssm(ssm_parallel);
    free_ssm(loaded_ssm);
    
    return 0;
}
