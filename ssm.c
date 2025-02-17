#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "ssm.h"

int main() {
    srand(time(NULL) ^ getpid());
    openblas_set_num_threads(4);

    // Parameters
    const int input_dim = 16;
    const int state_dim = 512;
    const int output_dim = 4;
    const int num_samples = 1024; // 32 * 32 samples
    const int seq_length = 32;  // Process sequences of 32 steps
    const int batch_size = 32;  // Process batches of 32 samples
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim, seq_length);
    
    // Initialize state space model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.001f;
    
    // Allocate memory for batch data
    float* batch_X = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* batch_y = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // Process data in sequences
        for (int seq_start = 0; seq_start <= num_samples - seq_length; seq_start += seq_length) {
            // Reset state at the beginning of each sequence
            memset(ssm->state, 0, ssm->batch_size * ssm->state_dim * sizeof(float));
            
            // Process sequence in batches
            for (int step = 0; step < seq_length; step += batch_size) {
                int current_batch_size = batch_size;
                if (seq_start + step + batch_size > num_samples) {
                    current_batch_size = num_samples - (seq_start + step);
                }
                
                // Prepare batch data
                for (int b = 0; b < current_batch_size; b++) {
                    int sample_idx = seq_start + step + b;
                    memcpy(&batch_X[b * input_dim], 
                           &X[sample_idx * input_dim], 
                           input_dim * sizeof(float));
                    memcpy(&batch_y[b * output_dim], 
                           &y[sample_idx * output_dim], 
                           output_dim * sizeof(float));
                }
                
                // Forward pass
                forward_pass(ssm, batch_X);
                
                // Calculate loss
                float loss = calculate_loss(ssm, batch_y);
                epoch_loss += loss;
                num_batches++;
                
                // Backward pass
                zero_gradients(ssm);
                backward_pass(ssm, batch_X);
                
                // Update weights
                update_weights(ssm, learning_rate);
            }
        }
        
        // Print progress
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", 
                   epoch + 1, num_epochs, epoch_loss / num_batches);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_ssm.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&now));

    // Save model and data
    save_model(ssm, model_fname);
    save_data_to_csv(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Verify saved model
    printf("\nVerifying saved model...\n");
    
    // Load the model back
    SSM* loaded_ssm = load_model(model_fname);
    
    float* all_predictions = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Generate all predictions
    for (int seq_start = 0; seq_start < num_samples; seq_start += batch_size) {
        int current_batch_size = batch_size;
        if (seq_start + batch_size > num_samples) {
            current_batch_size = num_samples - seq_start;
        }
        
        // Prepare batch data
        for (int b = 0; b < current_batch_size; b++) {
            memcpy(&batch_X[b * input_dim], 
                   &X[(seq_start + b) * input_dim], 
                   input_dim * sizeof(float));
        }
        
        // Forward pass
        forward_pass(loaded_ssm, batch_X);
        
        // Store predictions
        for (int b = 0; b < current_batch_size; b++) {
            memcpy(&all_predictions[(seq_start + b) * output_dim],
                   &loaded_ssm->predictions[b * output_dim],
                   output_dim * sizeof(float));
        }
    }
    
    printf("\nStatistics per output dimension:\n");
    printf("----------------------------------\n");
    
    double total_mse = 0.0;
    
    for (int dim = 0; dim < output_dim; dim++) {
        printf("\nDimension %d:\n", dim);
        
        // Calculate simple error statistics first
        double sum_sq_error = 0.0;
        double sum_abs_error = 0.0;
        double sum_actual = 0.0;
        double sum_pred = 0.0;
        
        for (int i = 0; i < num_samples; i++) {
            double pred = (double)all_predictions[i * output_dim + dim];
            double actual = (double)y[i * output_dim + dim];
            double error = pred - actual;
            
            if (!isnan(pred) && !isnan(actual)) {  // Add validity check
                sum_sq_error += error * error;
                sum_abs_error += fabs(error);
                sum_actual += actual;
                sum_pred += pred;
            }
        }
        
        double mse = sum_sq_error / num_samples;
        double mae = sum_abs_error / num_samples;
        double mean_actual = sum_actual / num_samples;
        double mean_pred = sum_pred / num_samples;
        
        printf("MSE: %.6f\n", mse);
        printf("MAE: %.6f\n", mae);
        printf("Mean actual: %.6f\n", mean_actual);
        printf("Mean predicted: %.6f\n", mean_pred);
        
        // Calculate R² with extra care
        double ss_tot = 0.0;
        double ss_res = 0.0;

        for (int i = 0; i < num_samples; i++) {
            double actual = (double)y[i * output_dim + dim];
            double pred = (double)all_predictions[i * output_dim + dim];
            
            if (!isnan(actual) && !isnan(pred)) {
                double diff_from_mean = actual - mean_actual;
                ss_tot += diff_from_mean * diff_from_mean;
                
                double residual = pred - actual;
                ss_res += residual * residual;
            }
        }

        double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
        printf("R²: %.6f\n", r2);
        
        total_mse += mse;
        
        // Print first 15 samples
        printf("\nFirst 15 samples:\n");
        printf("Predicted\tActual\t\tError\n");
        printf("----------------------------------------\n");
        for (int i = 0; i < 15 && i < num_samples; i++) {
            float pred = all_predictions[i * output_dim + dim];
            float actual = y[i * output_dim + dim];
            float error = pred - actual;
            if (!isnan(pred) && !isnan(actual)) {
                printf("%8.3f\t%8.3f\t%8.3f\n", pred, actual, error);
            }
        }
        printf("\n");
    }
    
    printf("\nOverall MSE: %.6f\n", total_mse / output_dim);

    // System stability analysis
    printf("\nSystem Stability Analysis:\n");
    float* v = (float*)malloc(state_dim * sizeof(float));
    float* Av = (float*)malloc(state_dim * sizeof(float));
    
    for (int i = 0; i < state_dim; i++) {
        v[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    
    float norm = 0.0f;
    for (int i = 0; i < state_dim; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrtf(norm);  // Use sqrtf for float
    for (int i = 0; i < state_dim; i++) {
        v[i] /= norm;
    }
    
    float spectral_radius = 0.0f;
    for (int iter = 0; iter < 20; iter++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    state_dim, state_dim,
                    1.0f, loaded_ssm->A, state_dim,
                    v, 1,
                    0.0f, Av, 1);
        
        norm = 0.0f;
        for (int i = 0; i < state_dim; i++) {
            norm += Av[i] * Av[i];
        }
        norm = sqrtf(norm);
        spectral_radius = norm;
        
        for (int i = 0; i < state_dim; i++) {
            v[i] = Av[i] / norm;
        }
    }
    
    printf("Estimated spectral radius of A: %.6f\n", spectral_radius);
    printf("System stability: %s\n", 
           spectral_radius < 1.0f ? "Stable" : "Potentially unstable");

    // Compute matrix norms
    float norm_A = 0.0f, norm_B = 0.0f, norm_C = 0.0f, norm_D = 0.0f;
    
    for (int i = 0; i < state_dim * state_dim; i++) {
        norm_A += loaded_ssm->A[i] * loaded_ssm->A[i];
    }
    for (int i = 0; i < state_dim * input_dim; i++) {
        norm_B += loaded_ssm->B[i] * loaded_ssm->B[i];
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        norm_C += loaded_ssm->C[i] * loaded_ssm->C[i];
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        norm_D += loaded_ssm->D[i] * loaded_ssm->D[i];
    }
    
    printf("\nFrobenius norms of matrices:\n");
    printf("||A|| = %.6f\n", sqrtf(norm_A));
    printf("||B|| = %.6f\n", sqrtf(norm_B));
    printf("||C|| = %.6f\n", sqrtf(norm_C));
    printf("||D|| = %.6f\n", sqrtf(norm_D));
    
    // Cleanup
    free(batch_X);
    free(batch_y);
    free(all_predictions);
    free(v);
    free(Av);
    free(X);
    free(y);
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    
    return 0;
}