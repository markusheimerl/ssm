#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "ssm.h"

int main() {
    srand(time(NULL));
    cudaSetDevice(0);

    // Parameters
    const int input_dim = 16;
    const int state_dim = 512;
    const int output_dim = 4;
    const int num_samples = 1024;
    const int batch_size = num_samples;
    
    // Generate synthetic data
    float *h_X, *h_y;
    generate_synthetic_data(&h_X, &h_y, num_samples, input_dim, output_dim);
    
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
    const float learning_rate = 0.001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset state at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, 
                             ssm->batch_size * ssm->state_dim * sizeof(float)));
        
        // Forward pass
        forward_pass(ssm, d_X);
        
        // Calculate loss
        float loss = calculate_loss(ssm, d_y);
        
        // Backward pass
        zero_gradients(ssm);
        backward_pass(ssm, d_X);
        
        // Update weights
        update_weights(ssm, learning_rate);
        
        // Print progress
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
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
    save_data_to_csv(h_X, h_y, num_samples, input_dim, output_dim, data_fname);
    
    // Verify saved model
    printf("\nVerifying saved model...\n");
    
    // Load the model back
    SSM* loaded_ssm = load_model(model_fname);
    
    // Reset state
    CHECK_CUDA(cudaMemset(loaded_ssm->d_state, 0, 
                         loaded_ssm->batch_size * loaded_ssm->state_dim * sizeof(float)));
    
    // Forward pass with loaded model
    forward_pass(loaded_ssm, d_X);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss(loaded_ssm, d_y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores
    printf("\nR² scores:\n");
    float* h_predictions = (float*)malloc(num_samples * output_dim * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_predictions, loaded_ssm->d_predictions,
                         num_samples * output_dim * sizeof(float), 
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            y_mean += h_y[j * output_dim + i];
        }
        y_mean /= num_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            float diff_res = h_y[j * output_dim + i] - h_predictions[j * output_dim + i];
            float diff_tot = h_y[j * output_dim + i] - y_mean;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        float r2 = 1.0f - (ss_res / ss_tot);
        printf("R² score for output y%d: %.8f\n", i, r2);
    }

    // Print sample predictions
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Output\t\tPredicted\tActual\t\tDifference\n");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < output_dim; i++) {
        printf("\ny%d:\n", i);
        for (int j = 0; j < 15; j++) {
            float pred = h_predictions[j * output_dim + i];
            float actual = h_y[j * output_dim + i];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", j, pred, actual, diff);
        }
        
        // Calculate MAE for this output
        float mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            mae += fabs(h_predictions[j * output_dim + i] - h_y[j * output_dim + i]);
        }
        mae /= num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }

    printf("\nSystem Stability Analysis:\n");
    
    // Copy matrix A to host for analysis
    float* h_A = (float*)malloc(state_dim * state_dim * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_A, loaded_ssm->d_A,
                         state_dim * state_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Using power iteration method for spectral radius
    float* v = (float*)malloc(state_dim * sizeof(float));
    float* Av = (float*)malloc(state_dim * sizeof(float));
    
    // Initialize random vector
    for (int i = 0; i < state_dim; i++) {
        v[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Normalize
    float norm = 0.0f;
    for (int i = 0; i < state_dim; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < state_dim; i++) {
        v[i] /= norm;
    }
    
    // Power iteration (20 iterations)
    float spectral_radius = 0.0f;
    for (int iter = 0; iter < 20; iter++) {
        // Compute Av
        for (int i = 0; i < state_dim; i++) {
            Av[i] = 0.0f;
            for (int j = 0; j < state_dim; j++) {
                Av[i] += h_A[i * state_dim + j] * v[j];
            }
        }
        
        // Compute new norm
        norm = 0.0f;
        for (int i = 0; i < state_dim; i++) {
            norm += Av[i] * Av[i];
        }
        norm = sqrt(norm);
        
        // Update spectral radius estimate
        spectral_radius = norm;
        
        // Normalize
        for (int i = 0; i < state_dim; i++) {
            v[i] = Av[i] / norm;
        }
    }
    
    printf("Estimated spectral radius of A: %.6f\n", spectral_radius);
    printf("System stability: %s\n", 
           spectral_radius < 1.0f ? "Stable" : "Potentially unstable");

    // Copy matrices to host for norm calculation
    float* h_B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* h_C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* h_D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_B, loaded_ssm->d_B,
                         state_dim * input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C, loaded_ssm->d_C,
                         output_dim * state_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, loaded_ssm->d_D,
                         output_dim * input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Compute matrix norms
    float norm_A = 0.0f, norm_B = 0.0f, norm_C = 0.0f, norm_D = 0.0f;
    
    for (int i = 0; i < state_dim * state_dim; i++) {
        norm_A += h_A[i] * h_A[i];
    }
    for (int i = 0; i < state_dim * input_dim; i++) {
        norm_B += h_B[i] * h_B[i];
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        norm_C += h_C[i] * h_C[i];
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        norm_D += h_D[i] * h_D[i];
    }
    
    printf("\nFrobenius norms of matrices:\n");
    printf("||A|| = %.6f\n", sqrt(norm_A));
    printf("||B|| = %.6f\n", sqrt(norm_B));
    printf("||C|| = %.6f\n", sqrt(norm_C));
    printf("||D|| = %.6f\n", sqrt(norm_D));
    
    // Cleanup
    free(h_X);
    free(h_y);
    free(h_predictions);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(v);
    free(Av);
    cudaFree(d_X);
    cudaFree(d_y);
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    
    return 0;
}