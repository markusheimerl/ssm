#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Error checking macro for cuBLAS calls
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

typedef struct {
    // State transition matrices (device pointers)
    float* d_A;  // state_dim x state_dim
    float* d_B;  // state_dim x input_dim
    float* d_C;  // output_dim x state_dim
    float* d_D;  // output_dim x input_dim

    // Host copies
    float* h_A;
    float* h_B;
    float* h_C;
    float* h_D;

    // Gradients (device pointers)
    float* d_A_grad;
    float* d_B_grad;
    float* d_C_grad;
    float* d_D_grad;

    // Adam parameters (device pointers)
    float* d_A_m;
    float* d_A_v;
    float* d_B_m;
    float* d_B_v;
    float* d_C_m;
    float* d_C_v;
    float* d_D_m;
    float* d_D_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;

    // Helper arrays (device pointers)
    float* d_state;          // batch_size x state_dim
    float* d_next_state;     // batch_size x state_dim
    float* d_pre_state;      // batch_size x state_dim
    float* d_predictions;    // batch_size x output_dim
    float* d_error;          // batch_size x output_dim
    float* d_state_error;    // batch_size x state_dim

    // Temporary buffers for matrix operations
    float* d_temp_state;     // batch_size x state_dim
    float* d_temp_output;    // batch_size x output_dim

    // cuBLAS handle
    cublasHandle_t cublas_handle;

    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int batch_size;
} SSM;

// CUDA kernel for swish activation forward pass
__global__ void swish_forward_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// CUDA kernel for swish activation backward pass
__global__ void swish_backward_kernel(float* grad_output, const float* input, 
                                    const float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float swish = output[idx];
        grad_output[idx] *= (swish + sigmoid * (1.0f - swish));
    }
}

// CUDA kernel for computing MSE loss
__global__ void mse_loss_kernel(float* error, const float* predictions, 
                               const float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - targets[idx];
    }
}

// CUDA kernel for Adam optimizer updates
__global__ void adam_update_kernel(float* W, const float* W_grad, float* M, float* V,
                                 float beta1, float beta2, float epsilon,
                                 float learning_rate, float weight_decay,
                                 float beta1_t, float beta2_t, int size, 
                                 float batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = W_grad[idx] / batch_size;  // Normalize by batch size
        
        // Update momentum and velocity
        M[idx] = beta1 * M[idx] + (1.0f - beta1) * grad;
        V[idx] = beta2 * V[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected moments
        float m_hat = M[idx] / (1.0f - beta1_t);
        float v_hat = V[idx] / (1.0f - beta2_t);
        
        // Update weights with L2 regularization
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) -
                 learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

SSM* init_ssm(int input_dim, int state_dim, int output_dim, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    
    // Store dimensions
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->batch_size = batch_size;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&ssm->cublas_handle));
    
    // Initialize Adam parameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->t = 0;
    ssm->weight_decay = 0.01f;
    
    // Allocate host matrices
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
        ssm->h_A[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_A;
    }
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->h_B[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_B;
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->h_C[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_C;
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->h_D[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_D;
    }
    
    // Allocate device memory for matrices
    CHECK_CUDA(cudaMalloc(&ssm->d_A, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ssm->d_A_m, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_v, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_m, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_v, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_m, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_v, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_m, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_v, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for helper arrays
    CHECK_CUDA(cudaMalloc(&ssm->d_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_next_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_pre_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error, batch_size * state_dim * sizeof(float)));
    
    // Allocate device memory for temporary buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_output, batch_size * output_dim * sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(ssm->d_A, ssm->h_A, state_dim * state_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, ssm->h_B, state_dim * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, ssm->h_C, output_dim * state_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, ssm->h_D, output_dim * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    
    return ssm;
}

void forward_pass_ssm(SSM* ssm, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Next state computation: pre_state = Ax + Bu
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->state_dim,
                            &alpha,
                            ssm->d_A, ssm->state_dim,
                            ssm->d_state, ssm->state_dim,
                            &beta,
                            ssm->d_pre_state, ssm->state_dim));
    
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim,
                            &alpha,
                            ssm->d_B, ssm->state_dim,
                            d_X, ssm->input_dim,
                            &alpha,
                            ssm->d_pre_state, ssm->state_dim));
    
    // Apply swish activation
    int block_size = 256;
    int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(
        ssm->d_next_state, ssm->d_pre_state, ssm->batch_size * ssm->state_dim);
    
    // Output computation: predictions = Cs + Du
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->state_dim,
                            &alpha,
                            ssm->d_C, ssm->output_dim,
                            ssm->d_next_state, ssm->state_dim,
                            &beta,
                            ssm->d_predictions, ssm->output_dim));
    
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->input_dim,
                            &alpha,
                            ssm->d_D, ssm->output_dim,
                            d_X, ssm->input_dim,
                            &alpha,
                            ssm->d_predictions, ssm->output_dim));
    
    // Update state
    CHECK_CUDA(cudaMemcpy(ssm->d_state, ssm->d_next_state,
                         ssm->batch_size * ssm->state_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
}

float calculate_loss_ssm(SSM* ssm, float* d_y) {
    int size = ssm->batch_size * ssm->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    // Compute error
    mse_loss_kernel<<<num_blocks, block_size>>>(
        ssm->d_error, ssm->d_predictions, d_y, size);
    
    // Compute loss (sum of squared errors)
    float loss;
    CHECK_CUBLAS(cublasSdot(ssm->cublas_handle, size,
                           ssm->d_error, 1,
                           ssm->d_error, 1,
                           &loss));
    
    return loss / size;
}

void zero_gradients_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float)));
}

void backward_pass_ssm(SSM* ssm, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Gradient for C: ∂L/∂C = error · next_state^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->output_dim, ssm->state_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_error, ssm->output_dim,
                            ssm->d_next_state, ssm->state_dim,
                            &beta,
                            ssm->d_C_grad, ssm->output_dim));
    
    // Gradient for D: ∂L/∂D = error · X^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->output_dim, ssm->input_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_error, ssm->output_dim,
                            d_X, ssm->input_dim,
                            &beta,
                            ssm->d_D_grad, ssm->output_dim));
    
    // State error: ∂L/∂next_state = C^T · error
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->output_dim,
                            &alpha,
                            ssm->d_C, ssm->output_dim,
                            ssm->d_error, ssm->output_dim,
                            &beta,
                            ssm->d_state_error, ssm->state_dim));
    
    // Apply swish backward
    int block_size = 256;
    int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(
        ssm->d_state_error, ssm->d_pre_state, ssm->d_next_state,
        ssm->batch_size * ssm->state_dim);
    
    // Gradient for A: ∂L/∂A = state_error · state^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->state_dim, ssm->state_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_state_error, ssm->state_dim,
                            ssm->d_state, ssm->state_dim,
                            &beta,
                            ssm->d_A_grad, ssm->state_dim));
    
    // Gradient for B: ∂L/∂B = state_error · X^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->state_dim, ssm->input_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_state_error, ssm->state_dim,
                            d_X, ssm->input_dim,
                            &beta,
                            ssm->d_B_grad, ssm->state_dim));
}

void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    
    int block_size = 256;
    
    // Update A
    int num_blocks_A = (ssm->state_dim * ssm->state_dim + block_size - 1) / block_size;
    adam_update_kernel<<<num_blocks_A, block_size>>>(
        ssm->d_A, ssm->d_A_grad, ssm->d_A_m, ssm->d_A_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        beta1_t, beta2_t, ssm->state_dim * ssm->state_dim, ssm->batch_size);
    
    // Update B
    int num_blocks_B = (ssm->state_dim * ssm->input_dim + block_size - 1) / block_size;
    adam_update_kernel<<<num_blocks_B, block_size>>>(
        ssm->d_B, ssm->d_B_grad, ssm->d_B_m, ssm->d_B_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        beta1_t, beta2_t, ssm->state_dim * ssm->input_dim, ssm->batch_size);
    
    // Update C
    int num_blocks_C = (ssm->output_dim * ssm->state_dim + block_size - 1) / block_size;
    adam_update_kernel<<<num_blocks_C, block_size>>>(
        ssm->d_C, ssm->d_C_grad, ssm->d_C_m, ssm->d_C_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        beta1_t, beta2_t, ssm->output_dim * ssm->state_dim, ssm->batch_size);
    
    // Update D
    int num_blocks_D = (ssm->output_dim * ssm->input_dim + block_size - 1) / block_size;
    adam_update_kernel<<<num_blocks_D, block_size>>>(
        ssm->d_D, ssm->d_D_grad, ssm->d_D_m, ssm->d_D_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        beta1_t, beta2_t, ssm->output_dim * ssm->input_dim, ssm->batch_size);
}

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
    cudaFree(ssm->d_A_m);
    cudaFree(ssm->d_A_v);
    cudaFree(ssm->d_B_m);
    cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m);
    cudaFree(ssm->d_C_v);
    cudaFree(ssm->d_D_m);
    cudaFree(ssm->d_D_v);
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

void save_ssm(SSM* ssm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
    // Allocate temporary host buffers
    size_t size_A = ssm->state_dim * ssm->state_dim * sizeof(float);
    size_t size_B = ssm->state_dim * ssm->input_dim * sizeof(float);
    size_t size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    size_t size_D = ssm->output_dim * ssm->input_dim * sizeof(float);
    
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_D);
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(h_A, ssm->d_A, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B, ssm->d_B, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C, ssm->d_C, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, ssm->d_D, size_D, cudaMemcpyDeviceToHost));
    
    // Save matrices
    fwrite(h_A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Save Adam state
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Copy and save Adam parameters
    float* h_A_m = (float*)malloc(size_A);
    float* h_A_v = (float*)malloc(size_A);
    float* h_B_m = (float*)malloc(size_B);
    float* h_B_v = (float*)malloc(size_B);
    float* h_C_m = (float*)malloc(size_C);
    float* h_C_v = (float*)malloc(size_C);
    float* h_D_m = (float*)malloc(size_D);
    float* h_D_v = (float*)malloc(size_D);
    
    CHECK_CUDA(cudaMemcpy(h_A_m, ssm->d_A_m, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_A_v, ssm->d_A_v, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_m, ssm->d_B_m, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_v, ssm->d_B_v, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_m, ssm->d_C_m, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_v, ssm->d_C_v, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_m, ssm->d_D_m, size_D, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_v, ssm->d_D_v, size_D, cudaMemcpyDeviceToHost));
    
    fwrite(h_A_m, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_A_v, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B_m, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_B_v, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(h_D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_A_m);
    free(h_A_v);
    free(h_B_m);
    free(h_B_v);
    free(h_C_m);
    free(h_C_v);
    free(h_D_m);
    free(h_D_v);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

SSM* load_ssm(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, state_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize new SSM
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Allocate temporary host buffers
    size_t size_A = state_dim * state_dim * sizeof(float);
    size_t size_B = state_dim * input_dim * sizeof(float);
    size_t size_C = output_dim * state_dim * sizeof(float);
    size_t size_D = output_dim * input_dim * sizeof(float);
    
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_D);
    
    // Read matrices
    fread(h_A, sizeof(float), state_dim * state_dim, file);
    fread(h_B, sizeof(float), state_dim * input_dim, file);
    fread(h_C, sizeof(float), output_dim * state_dim, file);
    fread(h_D, sizeof(float), output_dim * input_dim, file);
    
    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, h_C, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, h_D, size_D, cudaMemcpyHostToDevice));
    
    // Read Adam state
    fread(&ssm->t, sizeof(int), 1, file);
    
    // Read Adam parameters
    float* h_A_m = (float*)malloc(size_A);
    float* h_A_v = (float*)malloc(size_A);
    float* h_B_m = (float*)malloc(size_B);
    float* h_B_v = (float*)malloc(size_B);
    float* h_C_m = (float*)malloc(size_C);
    float* h_C_v = (float*)malloc(size_C);
    float* h_D_m = (float*)malloc(size_D);
    float* h_D_v = (float*)malloc(size_D);
    
    fread(h_A_m, sizeof(float), state_dim * state_dim, file);
    fread(h_A_v, sizeof(float), state_dim * state_dim, file);
    fread(h_B_m, sizeof(float), state_dim * input_dim, file);
    fread(h_B_v, sizeof(float), state_dim * input_dim, file);
    fread(h_C_m, sizeof(float), output_dim * state_dim, file);
    fread(h_C_v, sizeof(float), output_dim * state_dim, file);
    fread(h_D_m, sizeof(float), output_dim * input_dim, file);
    fread(h_D_v, sizeof(float), output_dim * input_dim, file);
    
    // Copy Adam parameters to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A_m, h_A_m, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_v, h_A_v, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_m, h_B_m, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_v, h_B_v, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_m, h_C_m, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_v, h_C_v, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m, h_D_m, size_D, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v, h_D_v, size_D, cudaMemcpyHostToDevice));
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_A_m);
    free(h_A_v);
    free(h_B_m);
    free(h_B_v);
    free(h_C_m);
    free(h_C_v);
    free(h_D_m);
    free(h_D_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}

#endif