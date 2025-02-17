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
    
    // Model Evaluation
    printf("\nModel Evaluation\n");
    printf("================\n");

    SSM* loaded_ssm = load_model(model_fname);
    float* predictions = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Generate predictions for all samples
    memset(loaded_ssm->state, 0, loaded_ssm->batch_size * loaded_ssm->state_dim * sizeof(float));
    
    for (int i = 0; i < num_samples; i += batch_size) {
        int current_batch_size = (i + batch_size > num_samples) ? (num_samples - i) : batch_size;
        
        // Prepare batch
        for (int b = 0; b < current_batch_size; b++) {
            memcpy(&batch_X[b * input_dim], &X[(i + b) * input_dim], input_dim * sizeof(float));
        }
        
        forward_pass(loaded_ssm, batch_X);
        
        // Store predictions
        for (int b = 0; b < current_batch_size; b++) {
            memcpy(&predictions[(i + b) * output_dim], 
                   &loaded_ssm->predictions[b * output_dim],
                   output_dim * sizeof(float));
        }
    }

    // Evaluation metrics per dimension
    printf("\nPer-Dimension Analysis:\n");
    printf("=====================\n");

    double total_mse = 0.0;
    
    for (int d = 0; d < output_dim; d++) {
        printf("\nDimension %d:\n", d);
        printf("--------------\n");
        
        // Calculate MSE and R²
        double sum_sq_error = 0.0;
        double sum_y = 0.0;
        int valid_samples = 0;
        
        for (int i = 0; i < num_samples; i++) {
            double y_i = y[i * output_dim + d];
            double yhat_i = predictions[i * output_dim + d];
            
            if (!isnan(y_i) && !isnan(yhat_i) && !isinf(y_i) && !isinf(yhat_i)) {
                sum_sq_error += (y_i - yhat_i) * (y_i - yhat_i);
                sum_y += y_i;
                valid_samples++;
            }
        }
        
        if (valid_samples == 0) {
            printf("No valid samples for evaluation!\n");
            continue;
        }

        double mean_y = sum_y / valid_samples;
        double mse = sum_sq_error / valid_samples;
        
        // Compute R-squared
        double ss_tot = 0.0;
        for (int i = 0; i < num_samples; i++) {
            double y_i = y[i * output_dim + d];
            if (!isnan(y_i) && !isinf(y_i)) {
                ss_tot += (y_i - mean_y) * (y_i - mean_y);
            }
        }
        double r_squared = 1.0 - (sum_sq_error / ss_tot);

        // Print metrics
        printf("MSE: %.6f\n", mse);
        printf("R²:  %.6f\n", r_squared);
        
        // Print sample predictions
        printf("\nSample Predictions (first 10):\n");
        printf("Actual\t\tPredicted\tError\n");
        printf("----------------------------------------\n");
        for (int i = 0; i < 10 && i < num_samples; i++) {
            double y_i = y[i * output_dim + d];
            double yhat_i = predictions[i * output_dim + d];
            if (!isnan(y_i) && !isnan(yhat_i) && !isinf(y_i) && !isinf(yhat_i)) {
                printf("%.6f\t%.6f\t%.6f\n", y_i, yhat_i, yhat_i - y_i);
            }
        }
        
        total_mse += mse;
    }

    printf("\nOverall Performance:\n");
    printf("===================\n");
    printf("Average MSE across dimensions: %.6f\n", total_mse / output_dim);

    // Cleanup
    free(predictions);
    free(batch_X);
    free(batch_y);
    free(X);
    free(y);
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    
    return 0;
}