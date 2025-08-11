#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../data.h"
#include "ssm.h"

// Reshape data from [batch][time][feature] to [time][batch][feature]
void reshape_and_convert_data_for_batch_processing(float* X, float* y, __half** X_reshaped, __half** y_reshaped, int num_sequences, int seq_len, int input_dim, int output_dim) {
    *X_reshaped = (__half*)malloc(seq_len * num_sequences * input_dim * sizeof(__half));
    *y_reshaped = (__half*)malloc(seq_len * num_sequences * output_dim * sizeof(__half));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            // Convert X data
            for (int i = 0; i < input_dim; i++) {
                int src_idx = b * seq_len * input_dim + t * input_dim + i;
                int dst_idx = t * num_sequences * input_dim + b * input_dim + i;
                (*X_reshaped)[dst_idx] = __float2half(X[src_idx]);
            }
            
            // Convert y data
            for (int i = 0; i < output_dim; i++) {
                int src_idx = b * seq_len * output_dim + t * output_dim + i;
                int dst_idx = t * num_sequences * output_dim + b * output_dim + i;
                (*y_reshaped)[dst_idx] = __float2half(y[src_idx]);
            }
        }
    }
}

// Compute R² score
float compute_r2_score(__half* predictions, __half* targets, int size) {
    // Compute mean of targets
    double y_mean = 0.0;
    for (int i = 0; i < size; i++) {
        y_mean += __half2float(targets[i]);
    }
    y_mean /= size;
    
    double ss_res = 0.0;
    double ss_tot = 0.0;
    
    for (int i = 0; i < size; i++) {
        float pred = __half2float(predictions[i]);
        float actual = __half2float(targets[i]);
        
        double diff_res = actual - pred;
        double diff_tot = actual - y_mean;
        
        ss_res += diff_res * diff_res;
        ss_tot += diff_tot * diff_tot;
    }
    
    return (float)(1.0 - (ss_res / ss_tot));
}

// Compute MAE
float compute_mae(__half* predictions, __half* targets, int size) {
    double mae = 0.0;
    for (int i = 0; i < size; i++) {
        float pred = __half2float(predictions[i]);
        float actual = __half2float(targets[i]);
        mae += fabs(pred - actual);
    }
    return (float)(mae / size);
}

int main() {
    srand(time(NULL));

    // Initialize cuBLAS handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // Parameters
    const int input_dim = 16;
    const int state_dim = 128;
    const int output_dim = 4;
    const int seq_len = 32;
    const int num_sequences = 32;
    const int batch_size = num_sequences;
    
    // Generate synthetic sequence data (FP32)
    float *X, *y;
    generate_synthetic_data(&X, &y, num_sequences, seq_len, input_dim, output_dim, -3.0f, 3.0f);
    
    // Reshape data for batch processing
    __half *X_reshaped, *y_reshaped;
    reshape_and_convert_data_for_batch_processing(X, y, &X_reshaped, &y_reshaped, num_sequences, seq_len, input_dim, output_dim);
    
    // Allocate device memory for input and output and copy data
    __half *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, seq_len * batch_size * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_y, seq_len * batch_size * output_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_X, X_reshaped, seq_len * batch_size * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y_reshaped, seq_len * batch_size * output_dim * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Initialize state space model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size, cublas_handle);
    
    // Training parameters
    const int num_epochs = 3000;
    const float learning_rate = 0.0003f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        reset_state_ssm(ssm);
        for (int t = 0; t < seq_len; t++) {
            __half* X_t = d_X + t * batch_size * input_dim;
            forward_pass_ssm(ssm, X_t, t);
        }
        
        // Calculate loss
        float loss = calculate_loss_ssm(ssm, d_y);

        // Print progress
        if (epoch > 0 && epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_ssm(ssm);
        for (int t = seq_len - 1; t >= 0; t--) {
            __half* X_t = d_X + t * batch_size * input_dim;
            backward_pass_ssm(ssm, X_t, t);
        }
        
        // Update weights
        update_weights_ssm(ssm, learning_rate);
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save model and data with timestamped filenames
    save_ssm(ssm, model_fname);
    save_data(X, y, num_sequences, seq_len, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    SSM* loaded_ssm = load_ssm(model_fname, batch_size, cublas_handle);
    
    // Forward pass with loaded model
    reset_state_ssm(loaded_ssm);
    for (int t = 0; t < seq_len; t++) {
        __half* X_t = d_X + t * batch_size * input_dim;
        forward_pass_ssm(loaded_ssm, X_t, t);
    }
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_ssm(loaded_ssm, d_y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Copy predictions from device to host
    __half* predictions = (__half*)malloc(seq_len * batch_size * output_dim * sizeof(__half));
    CHECK_CUDA(cudaMemcpy(predictions, loaded_ssm->d_layer2_output, seq_len * batch_size * output_dim * sizeof(__half), cudaMemcpyDeviceToHost));

    // Calculate R² scores
    printf("\nR² scores:\n");
    int total_samples = num_sequences * seq_len;
    for (int i = 0; i < output_dim; i++) {
        // Extract predictions and targets for this output dimension
        __half* output_predictions = (__half*)malloc(total_samples * sizeof(__half));
        __half* output_targets = (__half*)malloc(total_samples * sizeof(__half));
        
        int sample_idx = 0;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                output_predictions[sample_idx] = predictions[idx];
                output_targets[sample_idx] = y_reshaped[idx];
                sample_idx++;
            }
        }
        
        float r2 = compute_r2_score(output_predictions, output_targets, total_samples);
        printf("R² score for output y%d: %.8f\n", i, r2);
        
        free(output_predictions);
        free(output_targets);
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
            float pred = __half2float(predictions[idx]);
            float actual = __half2float(y_reshaped[idx]);
            float diff = pred - actual;
            printf("t=%d\t\t%8.3f\t%8.3f\t%8.3f\n", t, pred, actual, diff);
        }
        
        // Calculate MAE for this output across all sequences and time steps
        __half* output_predictions = (__half*)malloc(total_samples * sizeof(__half));
        __half* output_targets = (__half*)malloc(total_samples * sizeof(__half));
        
        int sample_idx = 0;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                output_predictions[sample_idx] = predictions[idx];
                output_targets[sample_idx] = y_reshaped[idx];
                sample_idx++;
            }
        }
        
        float mae = compute_mae(output_predictions, output_targets, total_samples);
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
        
        free(output_predictions);
        free(output_targets);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_reshaped);
    free(y_reshaped);
    free(predictions);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}