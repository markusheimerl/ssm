#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Error checking macros for CUDA and cuBLAS calls
// ---------------------------------------------------------------------
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---------------------------------------------------------------------
// Structure definition for the state-space model (SSM)
// This version uses cuBLAS and integrates natural gradient descent.
// ---------------------------------------------------------------------
typedef struct {
    // State transition matrices (stored on device)
    float* d_A;  // state_dim x state_dim
    float* d_B;  // state_dim x input_dim
    float* d_C;  // output_dim x state_dim
    float* d_D;  // output_dim x input_dim

    // Host copies for saving/loading the model
    float* h_A;
    float* h_B;
    float* h_C;
    float* h_D;

    // Gradients (device pointers)
    float* d_A_grad;
    float* d_B_grad;
    float* d_C_grad;
    float* d_D_grad;

    // Fisher Information Matrices (FIM) for natural gradient (device pointers)
    float* d_A_fim;
    float* d_B_fim;
    float* d_C_fim;
    float* d_D_fim;

    // Natural gradient hyper‐parameters
    float damping;       // e.g., 1e-4
    float weight_decay;  // e.g., 0.01

    // Helper arrays (device pointers)
    float* d_state;         // batch_size x state_dim
    float* d_next_state;    // batch_size x state_dim
    float* d_pre_state;     // batch_size x state_dim (pre-activation state)
    float* d_predictions;   // batch_size x output_dim
    float* d_error;         // batch_size x output_dim
    float* d_state_error;   // batch_size x state_dim

    // Temporary buffers for matrix operations
    float* d_temp_state;    // batch_size x state_dim
    float* d_temp_output;   // batch_size x output_dim

    // cuBLAS handle
    cublasHandle_t cublas_handle;

    // Dimensions of the network
    int input_dim;
    int state_dim;
    int output_dim;
    int batch_size;
} SSM;

// ---------------------------------------------------------------------
// CUDA kernel: swish activation forward pass
// swish(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------
__global__ void swish_forward_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: swish activation backward pass
// Computes derivative using: grad_output *= swish + sigmoid*(1-swish)
// ---------------------------------------------------------------------
__global__ void swish_backward_kernel(float* grad_output, const float* input, 
                                        const float* activated, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float swish = activated[idx];  // already computed activated value
        grad_output[idx] *= (swish + sigmoid * (1.0f - swish));
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Mean Squared Error loss computation (elementwise error)
// ---------------------------------------------------------------------
__global__ void mse_loss_kernel(float* error, const float* predictions, 
                                const float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        error[idx] = diff;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Natural Gradient update (per weight element)
// It updates FIM estimates and then uses them to calculate the natural gradient.
// Each weight is updated as:
//   fim[i] = 0.95 * fim[i] + 0.05 * (grad_i/batch_size)^2
//   nat = grad[i] / (sqrt(fim[i] + damping) + 1e-8)
//   W[i] = W[i]*(1 - learning_rate*weight_decay) - (learning_rate/n_batch)*nat
// ---------------------------------------------------------------------
__global__ void natural_gradient_update_kernel(float* W, const float* grad, float* fim, 
                                                int size, float damping, float weight_decay, 
                                                float learning_rate, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / ((float) batch_size);
        // Update running average of the Fisher Information Matrix (diagonal approximation)
        fim[idx] = 0.95f * fim[idx] + 0.05f * (g * g);
        // Compute natural gradient (elementwise)
        float nat = grad[idx] / (sqrtf(fim[idx] + damping) + 1e-8f);
        // Update weight with L2 weight decay
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) - learning_rate * nat / ((float) batch_size);
    }
}

// ---------------------------------------------------------------------
// Function: init_ssm
// Initializes the SSM structure, allocates host and device memory,
// sets initial weights with scaled random values, and copies them to device.
// ---------------------------------------------------------------------
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->batch_size = batch_size;

    // Set natural gradient hyperparameters
    ssm->damping = 1e-4f;
    ssm->weight_decay = 0.01f;

    // Create cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&ssm->cublas_handle));

    // Allocate host memory for weight matrices
    ssm->h_A = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->h_B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->h_C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->h_D = (float*)malloc(output_dim * input_dim * sizeof(float));

    // Initialize matrices with scaled random values
    float scale_A = 1.0f / sqrtf(state_dim);
    float scale_B = 1.0f / sqrtf(input_dim);
    float scale_C = 1.0f / sqrtf(state_dim);
    float scale_D = 1.0f / sqrtf(input_dim);

    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->h_A[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_A;
    }
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->h_B[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_B;
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->h_C[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_C;
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->h_D[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_D;
    }

    // Allocate device memory for weight matrices
    CHECK_CUDA(cudaMalloc(&ssm->d_A, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(float)));

    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, output_dim * input_dim * sizeof(float)));

    // Allocate device memory for Fisher Information Matrices (FIM) and set to zero
    CHECK_CUDA(cudaMalloc(&ssm->d_A_fim, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_fim, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_fim, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_fim, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_fim, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_fim, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_fim, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_fim, 0, output_dim * input_dim * sizeof(float)));

    // Allocate helper arrays
    CHECK_CUDA(cudaMalloc(&ssm->d_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_next_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_pre_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error, batch_size * state_dim * sizeof(float)));

    // Allocate temporary buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_output, batch_size * output_dim * sizeof(float)));

    // Copy weight matrices from host to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, ssm->h_A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, ssm->h_B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, ssm->h_C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, ssm->h_D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));

    return ssm;
}

// ---------------------------------------------------------------------
// Function: forward_pass
// Computes the forward pass:
//   pre_state = A * state + B * X
//   next_state = swish(pre_state)
//   predictions = C * next_state + D * X
// Updates the internal state to next_state.
// ---------------------------------------------------------------------
void forward_pass(SSM* ssm, float* d_X) {
    const float alpha = 1.0f, beta = 0.0f;

    // Compute pre_state = A * state
    // Dimensions: (state_dim x state_dim) * (state_dim x batch_size)
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->state_dim, ssm->batch_size, ssm->state_dim,
                             &alpha,
                             ssm->d_A, ssm->state_dim,
                             ssm->d_state, ssm->state_dim,
                             &beta,
                             ssm->d_pre_state, ssm->state_dim));

    // Add input contribution: pre_state += B * X
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->state_dim, ssm->batch_size, ssm->input_dim,
                             &alpha,
                             ssm->d_B, ssm->state_dim,
                             d_X, ssm->input_dim,
                             &alpha,
                             ssm->d_pre_state, ssm->state_dim));

    // Apply swish activation: next_state = swish(pre_state)
    int total_state = ssm->batch_size * ssm->state_dim;
    int block_size = 256;
    int num_blocks = (total_state + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(ssm->d_next_state, ssm->d_pre_state, total_state);
    
    // Compute predictions = C * next_state
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->output_dim, ssm->batch_size, ssm->state_dim,
                             &alpha,
                             ssm->d_C, ssm->output_dim,
                             ssm->d_next_state, ssm->state_dim,
                             &beta,
                             ssm->d_predictions, ssm->output_dim));
    // Add direct feedthrough: predictions += D * X
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->output_dim, ssm->batch_size, ssm->input_dim,
                             &alpha,
                             ssm->d_D, ssm->output_dim,
                             d_X, ssm->input_dim,
                             &alpha,
                             ssm->d_predictions, ssm->output_dim));
    
    // Update internal state: state = next_state
    CHECK_CUDA(cudaMemcpy(ssm->d_state, ssm->d_next_state,
                          ssm->batch_size * ssm->state_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------
// Function: calculate_loss
// Computes the Mean Squared Error loss between predictions and targets.
// ---------------------------------------------------------------------
float calculate_loss(SSM* ssm, float* d_y) {
    int size = ssm->batch_size * ssm->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    // Compute elementwise error: error = predictions - targets
    mse_loss_kernel<<<num_blocks, block_size>>>(ssm->d_error, ssm->d_predictions, d_y, size);

    // Compute sum of squared errors with cuBLAS dot product
    float loss = 0.0f;
    CHECK_CUBLAS(cublasSdot(ssm->cublas_handle, size,
                            ssm->d_error, 1,
                            ssm->d_error, 1,
                            &loss));
    return loss / size;
}

// ---------------------------------------------------------------------
// Function: zero_gradients
// Clears the gradient arrays on the device.
// ---------------------------------------------------------------------
void zero_gradients(SSM* ssm) {
    int size_A = ssm->state_dim * ssm->state_dim * sizeof(float);
    int size_B = ssm->state_dim * ssm->input_dim * sizeof(float);
    int size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    int size_D = ssm->output_dim * ssm->input_dim * sizeof(float);
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, size_A));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, size_B));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, size_C));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, size_D));
}

// ---------------------------------------------------------------------
// Function: backward_pass
// Computes gradients through the network using the chain rule:
//   dC_grad = error * (next_state)^T
//   dD_grad = error * (input)^T
//   state_error = C^T * error (back-propagated through output)
//   Then applies swish backward to state_error and computes gradients for A and B.
// ---------------------------------------------------------------------
void backward_pass(SSM* ssm, float* d_X) {
    const float alpha = 1.0f, beta = 0.0f;
    // Gradient for C: d_C_grad = error * (next_state)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->output_dim, ssm->state_dim, ssm->batch_size,
                             &alpha,
                             ssm->d_error, ssm->output_dim,
                             ssm->d_next_state, ssm->state_dim,
                             &beta,
                             ssm->d_C_grad, ssm->output_dim));
    // Gradient for D: d_D_grad = error * (X)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->output_dim, ssm->input_dim, ssm->batch_size,
                             &alpha,
                             ssm->d_error, ssm->output_dim,
                             d_X, ssm->input_dim,
                             &beta,
                             ssm->d_D_grad, ssm->output_dim));
    // Compute state error: state_error = C^T * error
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             ssm->state_dim, ssm->batch_size, ssm->output_dim,
                             &alpha,
                             ssm->d_C, ssm->output_dim,
                             ssm->d_error, ssm->output_dim,
                             &beta,
                             ssm->d_state_error, ssm->state_dim));
    // Apply swish backward: modify state_error in place
    int total_state = ssm->batch_size * ssm->state_dim;
    int block_size = 256;
    int num_blocks = (total_state + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(ssm->d_state_error, 
                                                      ssm->d_pre_state, 
                                                      ssm->d_next_state, 
                                                      total_state);
    // Gradient for A: d_A_grad = state_error * (state)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->state_dim, ssm->state_dim, ssm->batch_size,
                             &alpha,
                             ssm->d_state_error, ssm->state_dim,
                             ssm->d_state, ssm->state_dim,
                             &beta,
                             ssm->d_A_grad, ssm->state_dim));
    // Gradient for B: d_B_grad = state_error * (X)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->state_dim, ssm->input_dim, ssm->batch_size,
                             &alpha,
                             ssm->d_state_error, ssm->state_dim,
                             d_X, ssm->input_dim,
                             &beta,
                             ssm->d_B_grad, ssm->state_dim));
}

// ---------------------------------------------------------------------
// Function: update_weights
// Uses natural gradient descent to update each weight matrix.
// For each element it updates the corresponding Fisher Information Matrix
// using a running average and then adjusts the weight.
// ---------------------------------------------------------------------
void update_weights(SSM* ssm, float learning_rate) {
    int block_size = 256;
    int size_A = ssm->state_dim * ssm->state_dim;
    int size_B = ssm->state_dim * ssm->input_dim;
    int size_C = ssm->output_dim * ssm->state_dim;
    int size_D = ssm->output_dim * ssm->input_dim;

    int num_blocks_A = (size_A + block_size - 1) / block_size;
    int num_blocks_B = (size_B + block_size - 1) / block_size;
    int num_blocks_C = (size_C + block_size - 1) / block_size;
    int num_blocks_D = (size_D + block_size - 1) / block_size;

    natural_gradient_update_kernel<<<num_blocks_A, block_size>>>(
        ssm->d_A, ssm->d_A_grad, ssm->d_A_fim, size_A, 
        ssm->damping, ssm->weight_decay, learning_rate, ssm->batch_size);

    natural_gradient_update_kernel<<<num_blocks_B, block_size>>>(
        ssm->d_B, ssm->d_B_grad, ssm->d_B_fim, size_B, 
        ssm->damping, ssm->weight_decay, learning_rate, ssm->batch_size);

    natural_gradient_update_kernel<<<num_blocks_C, block_size>>>(
        ssm->d_C, ssm->d_C_grad, ssm->d_C_fim, size_C, 
        ssm->damping, ssm->weight_decay, learning_rate, ssm->batch_size);

    natural_gradient_update_kernel<<<num_blocks_D, block_size>>>(
        ssm->d_D, ssm->d_D_grad, ssm->d_D_fim, size_D, 
        ssm->damping, ssm->weight_decay, learning_rate, ssm->batch_size);
}

// ---------------------------------------------------------------------
// Function: free_ssm
// Frees all allocated memory (both device and host) and destroys the cuBLAS handle.
// ---------------------------------------------------------------------
void free_ssm(SSM* ssm) {
    // Free device memory
    cudaFree(ssm->d_A);
    cudaFree(ssm->d_B);
    cudaFree(ssm->d_C);
    cudaFree(ssm->d_D);
    cudaFree(ssm->d_A_grad);
    cudaFree(ssm->d_B_grad);
    cudaFree(ssm->d_C_grad);
    cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_A_fim);
    cudaFree(ssm->d_B_fim);
    cudaFree(ssm->d_C_fim);
    cudaFree(ssm->d_D_fim);
    cudaFree(ssm->d_state);
    cudaFree(ssm->d_next_state);
    cudaFree(ssm->d_pre_state);
    cudaFree(ssm->d_predictions);
    cudaFree(ssm->d_error);
    cudaFree(ssm->d_state_error);
    cudaFree(ssm->d_temp_state);
    cudaFree(ssm->d_temp_output);

    // Free host memory
    free(ssm->h_A);
    free(ssm->h_B);
    free(ssm->h_C);
    free(ssm->h_D);

    // Destroy cuBLAS handle
    cublasDestroy(ssm->cublas_handle);

    free(ssm);
}

// ---------------------------------------------------------------------
// Function: save_model
// Saves the model weights and FIM data to a binary file.
// ---------------------------------------------------------------------
void save_model(SSM* ssm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }

    // Write dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);

    size_t size_A = ssm->state_dim * ssm->state_dim * sizeof(float);
    size_t size_B = ssm->state_dim * ssm->input_dim * sizeof(float);
    size_t size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    size_t size_D = ssm->output_dim * ssm->input_dim * sizeof(float);

    // Allocate temporary host buffers
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_D);

    // Copy weight matrices from device to host
    CHECK_CUDA(cudaMemcpy(h_A, ssm->d_A, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B, ssm->d_B, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C, ssm->d_C, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, ssm->d_D, size_D, cudaMemcpyDeviceToHost));

    // Write weight matrices to file
    fwrite(h_A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D, sizeof(float), ssm->output_dim * ssm->input_dim, file);

    // Save FIM matrices
    float* h_A_fim = (float*)malloc(size_A);
    float* h_B_fim = (float*)malloc(size_B);
    float* h_C_fim = (float*)malloc(size_C);
    float* h_D_fim = (float*)malloc(size_D);
    CHECK_CUDA(cudaMemcpy(h_A_fim, ssm->d_A_fim, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_fim, ssm->d_B_fim, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_fim, ssm->d_C_fim, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_fim, ssm->d_D_fim, size_D, cudaMemcpyDeviceToHost));
    fwrite(h_A_fim, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B_fim, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_C_fim, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D_fim, sizeof(float), ssm->output_dim * ssm->input_dim, file);

    // Cleanup temporary host buffers
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_A_fim);
    free(h_B_fim);
    free(h_C_fim);
    free(h_D_fim);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// ---------------------------------------------------------------------
// Function: load_model
// Loads the model weights and FIM data from a binary file and initializes a new SSM.
// ---------------------------------------------------------------------
SSM* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }

    int input_dim, state_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);

    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);

    size_t size_A = state_dim * state_dim * sizeof(float);
    size_t size_B = state_dim * input_dim * sizeof(float);
    size_t size_C = output_dim * state_dim * sizeof(float);
    size_t size_D = output_dim * input_dim * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_D);

    fread(h_A, sizeof(float), state_dim * state_dim, file);
    fread(h_B, sizeof(float), state_dim * input_dim, file);
    fread(h_C, sizeof(float), output_dim * state_dim, file);
    fread(h_D, sizeof(float), output_dim * input_dim, file);

    CHECK_CUDA(cudaMemcpy(ssm->d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, h_C, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, h_D, size_D, cudaMemcpyHostToDevice));

    // Read FIM matrices and copy to device
    float* h_A_fim = (float*)malloc(size_A);
    float* h_B_fim = (float*)malloc(size_B);
    float* h_C_fim = (float*)malloc(size_C);
    float* h_D_fim = (float*)malloc(size_D);
    fread(h_A_fim, sizeof(float), state_dim * state_dim, file);
    fread(h_B_fim, sizeof(float), state_dim * input_dim, file);
    fread(h_C_fim, sizeof(float), output_dim * state_dim, file);
    fread(h_D_fim, sizeof(float), output_dim * input_dim, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_A_fim, h_A_fim, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_fim, h_B_fim, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_fim, h_C_fim, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_fim, h_D_fim, size_D, cudaMemcpyHostToDevice));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_A_fim);
    free(h_B_fim);
    free(h_C_fim);
    free(h_D_fim);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    return ssm;
}

#endif