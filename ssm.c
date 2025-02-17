#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "ssm.h"

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int input_dim = 16;
    const int state_dim = 512;
    const int output_dim = 4;
    const int num_samples = 1024;
    const int batch_size = num_samples;
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);
    
    // Initialize state space model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset state at the beginning of each epoch
        memset(ssm->state, 0, ssm->batch_size * ssm->state_dim * sizeof(float));
        
        // Forward pass
        forward_pass(ssm, X);
        
        // Calculate loss
        float loss = calculate_loss(ssm, y);
        
        // Backward pass
        zero_gradients(ssm);
        backward_pass(ssm, X);
        
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
    save_data_to_csv(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Verify saved model
    printf("\nVerifying saved model...\n");
    
    // Load the model back
    SSM* loaded_ssm = load_model(model_fname);
    
    // Reset state
    memset(loaded_ssm->state, 0, loaded_ssm->batch_size * loaded_ssm->state_dim * sizeof(float));
    
    // Forward pass with loaded model
    forward_pass(loaded_ssm, X);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss(loaded_ssm, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores
    printf("\nR² scores:\n");
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            y_mean += y[j * output_dim + i];
        }
        y_mean /= num_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            float diff_res = y[j * output_dim + i] - loaded_ssm->predictions[j * output_dim + i];
            float diff_tot = y[j * output_dim + i] - y_mean;
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
            float pred = loaded_ssm->predictions[j * output_dim + i];
            float actual = y[j * output_dim + i];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", j, pred, actual, diff);
        }
        
        // Calculate MAE for this output
        float mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            mae += fabs(loaded_ssm->predictions[j * output_dim + i] - y[j * output_dim + i]);
        }
        mae /= num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }

    // Additional analysis: System stability check
    printf("\nSystem Stability Analysis:\n");
    
    // Compute the spectral radius of A (maximum absolute eigenvalue)
    // Using a simple power iteration method
    float* v = (float*)malloc(state_dim * sizeof(float));
    float* Av = (float*)malloc(state_dim * sizeof(float));
    
    // Initialize random vector
    for (int i = 0; i < state_dim; i++) {
        v[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
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
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    state_dim, state_dim,
                    1.0f, loaded_ssm->A, state_dim,
                    v, 1,
                    0.0f, Av, 1);
        
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
    printf("||A|| = %.6f\n", sqrt(norm_A));
    printf("||B|| = %.6f\n", sqrt(norm_B));
    printf("||C|| = %.6f\n", sqrt(norm_C));
    printf("||D|| = %.6f\n", sqrt(norm_D));
    
    // Cleanup
    free(v);
    free(Av);
    free(X);
    free(y);
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    
    return 0;
}