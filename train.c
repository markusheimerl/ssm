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
    const int state_dim = 128;
    const int output_dim = 4;
    const int seq_len = 64;
    const int num_sequences = 64;
    const int batch_size = num_sequences;
    
    // Generate linear sequence data more suitable for linear SSM
    float *X, *y;
    printf("Generating linear sequence data for linear SSM...\n");
    generate_linear_sequence_data(&X, &y, num_sequences, seq_len, input_dim, output_dim);
    
    // Reshape data for batch processing
    float *X_reshaped, *y_reshaped;
    reshape_data_for_batch_processing(X, y, &X_reshaped, &y_reshaped,
                                    num_sequences, seq_len, input_dim, output_dim);
    
    // Initialize state space model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.0005f;
    
    printf("Training linear state space model...\n");
    printf("Architecture: %d -> %d -> %d (fully linear)\n", input_dim, state_dim, output_dim);
    printf("Sequence length: %d, Batch size: %d\n", seq_len, batch_size);
    printf("Epochs: %d, Learning rate: %.6f\n\n", num_epochs, learning_rate);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        forward_pass_ssm(ssm, X_reshaped);
        
        // Calculate loss
        float loss = calculate_loss_ssm(ssm, y_reshaped);

        // Print progress
        if (epoch > 0 && epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_ssm(ssm);
        backward_pass_ssm(ssm, X_reshaped);
        
        // Update weights
        update_weights_ssm(ssm, learning_rate);
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_linear_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_linear_data.csv", 
             localtime(&now));

    // Save model and data with timestamped filenames
    save_ssm(ssm, model_fname);
    save_synthetic_sequence_data_to_csv(X, y, num_sequences, seq_len, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    SSM* loaded_ssm = load_ssm(model_fname, batch_size);
    
    // Forward pass with loaded model
    forward_pass_ssm(loaded_ssm, X_reshaped);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_ssm(loaded_ssm, y_reshaped);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating linear model performance...\n");

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
    
    printf("\nNote: This linear SSM should perform better on linear data compared to complex synthetic data.\n");
    printf("The model uses purely linear dynamics: H_t = X_t B^T + H_{t-1} A^T, Y_t = H_t C^T + X_t D^T\n");
    
    // Cleanup
    free(X);
    free(y);
    free(X_reshaped);
    free(y_reshaped);
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    
    return 0;
}