#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "../../data.h"
#include "ssm.h"

int main() {
    srand(time(NULL) ^ getpid());

    // Parameters
    const int input_dim = 16;
    const int state_dim = 512;
    const int output_dim = 4;
    const int seq_length = 32;
    const int batch_size = 32;
    const int num_samples = seq_length * batch_size;
    
    // Generate synthetic data
    float *h_X, *h_y;
    generate_synthetic_data(&h_X, &h_y, num_samples, input_dim, output_dim, seq_length);
    
    // Transfer data to GPU
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, num_samples * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, num_samples * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, num_samples * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, num_samples * output_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Initialize state space model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.0001f;
    
    // Allocate memory for batch data on GPU
    float *d_batch_X, *d_batch_y;
    CHECK_CUDA(cudaMalloc(&d_batch_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_y, batch_size * output_dim * sizeof(float)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // Process data in sequences
        for (int seq_start = 0; seq_start <= num_samples - seq_length; seq_start += seq_length) {
            // Reset state at the beginning of each sequence
            CHECK_CUDA(cudaMemset(ssm->d_state1, 0, 
                                 ssm->batch_size * ssm->state_dim * sizeof(float)));
            CHECK_CUDA(cudaMemset(ssm->d_state2, 0, 
                                 ssm->batch_size * ssm->state_dim * sizeof(float)));
            
            // Process sequence in batches
            for (int step = 0; step < seq_length; step += batch_size) {
                int current_batch_size = batch_size;
                if (seq_start + step + batch_size > num_samples) {
                    current_batch_size = num_samples - (seq_start + step);
                }
                
                // Prepare batch data
                CHECK_CUDA(cudaMemcpy(d_batch_X, 
                                    &d_X[(seq_start + step) * input_dim],
                                    current_batch_size * input_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
                CHECK_CUDA(cudaMemcpy(d_batch_y,
                                    &d_y[(seq_start + step) * output_dim],
                                    current_batch_size * output_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
                
                // Forward pass
                forward_pass(ssm, d_batch_X);
                
                // Calculate loss
                float loss = calculate_loss(ssm, d_batch_y);
                epoch_loss += loss;
                num_batches++;
                
                // Backward pass
                zero_gradients(ssm);
                backward_pass(ssm, d_batch_X);
                
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
    save_ssm(ssm, model_fname);
    save_data_to_csv(h_X, h_y, num_samples, input_dim, output_dim, data_fname);
    
    // Model Evaluation
    printf("\nModel Evaluation\n");
    printf("================\n");

    SSM* loaded_ssm = load_ssm(model_fname);
    float* h_predictions = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Generate predictions for all samples
    CHECK_CUDA(cudaMemset(loaded_ssm->d_state1, 0,
                         loaded_ssm->batch_size * loaded_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(loaded_ssm->d_state2, 0,
                         loaded_ssm->batch_size * loaded_ssm->state_dim * sizeof(float)));
    
    // Process sequences for prediction
    for (int seq_start = 0; seq_start <= num_samples - seq_length; seq_start += seq_length) {
        // Reset state for each sequence
        CHECK_CUDA(cudaMemset(loaded_ssm->d_state1, 0,
                             loaded_ssm->batch_size * loaded_ssm->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(loaded_ssm->d_state2, 0,
                             loaded_ssm->batch_size * loaded_ssm->state_dim * sizeof(float)));
        
        // Process sequence
        for (int step = 0; step < seq_length; step += batch_size) {
            int current_batch_size = batch_size;
            if (seq_start + step + batch_size > num_samples) {
                current_batch_size = num_samples - (seq_start + step);
            }
            
            // Prepare batch
            CHECK_CUDA(cudaMemcpy(d_batch_X, 
                                &d_X[(seq_start + step) * input_dim],
                                current_batch_size * input_dim * sizeof(float),
                                cudaMemcpyDeviceToDevice));
            
            // Forward pass
            forward_pass(loaded_ssm, d_batch_X);
            
            // Copy predictions to host
            CHECK_CUDA(cudaMemcpy(&h_predictions[(seq_start + step) * output_dim],
                                loaded_ssm->d_predictions,
                                current_batch_size * output_dim * sizeof(float),
                                cudaMemcpyDeviceToHost));
        }
    }

    // Evaluation metrics per dimension
    printf("\nPer-Dimension Analysis:\n");
    printf("=====================\n");

    double total_mse = 0.0;
    
    for (int d = 0; d < output_dim; d++) {
        printf("\nDimension %d:\n", d);
        printf("--------------\n");
        
        // Calculate statistics
        double sum_sq_error = 0.0;
        double sum_y = 0.0;
        double sum_sq_y = 0.0;
        double sum_yhat = 0.0;
        double sum_sq_yhat = 0.0;
        double sum_y_yhat = 0.0;
        int valid_samples = 0;
        
        // First pass: calculate sums
        for (int i = 0; i < num_samples; i++) {
            double y_i = h_y[i * output_dim + d];
            double yhat_i = h_predictions[i * output_dim + d];
            
            if (!isnan(y_i) && !isnan(yhat_i) && !isinf(y_i) && !isinf(yhat_i)) {
                sum_sq_error += (y_i - yhat_i) * (y_i - yhat_i);
                sum_y += y_i;
                sum_sq_y += y_i * y_i;
                sum_yhat += yhat_i;
                sum_sq_yhat += yhat_i * yhat_i;
                sum_y_yhat += y_i * yhat_i;
                valid_samples++;
            }
        }
        
        if (valid_samples == 0) {
            printf("No valid samples for evaluation!\n");
            continue;
        }

        // Calculate statistics
        double mean_y = sum_y / valid_samples;
        double mean_yhat = sum_yhat / valid_samples;
        double mse = sum_sq_error / valid_samples;
        
        // Calculate R-squared components
        double ss_tot = sum_sq_y - (sum_y * sum_y / valid_samples);
        double ss_res = sum_sq_error;
        
        // Calculate correlation coefficient
        double numerator = sum_y_yhat - (sum_y * sum_yhat / valid_samples);
        double denominator = sqrt((sum_sq_y - (sum_y * sum_y / valid_samples)) * 
                                (sum_sq_yhat - (sum_yhat * sum_yhat / valid_samples)));
        double correlation = (denominator > 1e-10) ? numerator / denominator : 0.0;
        
        // Calculate R-squared
        double r_squared = (ss_tot > 1e-10) ? 1.0 - (ss_res / ss_tot) : 0.0;

        // Print metrics
        printf("MSE: %.6f\n", mse);
        printf("R²:  %.6f\n", r_squared);
        printf("Correlation: %.6f\n", correlation);
        printf("Mean Actual: %.6f\n", mean_y);
        printf("Mean Predicted: %.6f\n", mean_yhat);
        
        // Print sample predictions
        printf("\nSample Predictions (first 10):\n");
        printf("Actual\t\tPredicted\tError\n");
        printf("----------------------------------------\n");
        for (int i = 0; i < 10 && i < num_samples; i++) {
            double y_i = h_y[i * output_dim + d];
            double yhat_i = h_predictions[i * output_dim + d];
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
    free(h_X);
    free(h_y);
    free(h_predictions);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_batch_X);
    cudaFree(d_batch_y);
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    
    return 0;
}