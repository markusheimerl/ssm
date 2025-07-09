#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLAS Error checking macro
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Device pointers for state space matrices
    float* d_A;           // state_dim x state_dim (state transition)
    float* d_B;           // state_dim x input_dim (input to state)
    float* d_C;           // output_dim x state_dim (state to output)
    float* d_D;           // output_dim x input_dim (input to output)
    
    // Device pointers for gradients
    float* d_A_grad;      // state_dim x state_dim
    float* d_B_grad;      // state_dim x input_dim
    float* d_C_grad;      // output_dim x state_dim
    float* d_D_grad;      // output_dim x input_dim
    
    // Device pointers for Adam parameters
    float* d_A_m; float* d_A_v;
    float* d_B_m; float* d_B_v;
    float* d_C_m; float* d_C_v;
    float* d_D_m; float* d_D_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Device pointers for helper arrays (time-major format)
    float* d_states;          // seq_len x batch_size x state_dim
    float* d_predictions;     // seq_len x batch_size x output_dim
    float* d_error;          // seq_len x batch_size x output_dim
    float* d_state_error;    // seq_len x batch_size x state_dim
    float* d_state_outputs;  // seq_len x batch_size x state_dim
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} SSM;

// Initialize the state space model
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    
    // Store dimensions
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->seq_len = seq_len;
    ssm->batch_size = batch_size;
    
    // Initialize Adam parameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->t = 0;
    ssm->weight_decay = 0.001f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&ssm->cublas_handle));
    
    // Allocate host memory for initialization
    float* A = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Initialize matrices with careful scaling for stability
    float scale_B = 0.5f / sqrt(input_dim);
    float scale_C = 0.5f / sqrt(state_dim);
    float scale_D = 0.1f / sqrt(input_dim);
    
    // Initialize A as a stable matrix with eigenvalues < 1
    for (int i = 0; i < state_dim * state_dim; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * 0.05f;
    }
    
    // Add diagonal stability
    for (int i = 0; i < state_dim; i++) {
        A[i * state_dim + i] = 0.5f + ((float)rand() / (float)RAND_MAX * 0.3f);
    }
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        B[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        C[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        D[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_D;
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
    CHECK_CUDA(cudaMalloc(&ssm->d_states, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_outputs, seq_len * batch_size * state_dim * sizeof(float)));
    
    // Copy initialized matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    
    // Free host memory
    free(A);
    free(B);
    free(C);
    free(D);
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    // Free device memory
    cudaFree(ssm->d_A); cudaFree(ssm->d_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);
    cudaFree(ssm->d_A_grad); cudaFree(ssm->d_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_A_m); cudaFree(ssm->d_A_v); cudaFree(ssm->d_B_m); cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m); cudaFree(ssm->d_C_v); cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);
    cudaFree(ssm->d_states); cudaFree(ssm->d_predictions); cudaFree(ssm->d_error); 
    cudaFree(ssm->d_state_error); cudaFree(ssm->d_state_outputs);
    
    // Destroy cuBLAS handle
    cublasDestroy(ssm->cublas_handle);
    
    free(ssm);
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel_ssm(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel_ssm(float* grad_input, float* grad_output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        grad_input[idx] = grad_output[idx] * sigmoid * (1.0f + x * (1.0f - sigmoid));
    }
}

// CUDA kernel for calculating error
__global__ void calc_error_kernel_ssm(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

// Forward pass
void forward_pass_ssm(SSM* ssm, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Clear states
    CHECK_CUDA(cudaMemset(ssm->d_states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
    
    for (int t = 0; t < ssm->seq_len; t++) {
        // Pointers to current timestep data
        float* d_X_t = d_X + t * ssm->batch_size * ssm->input_dim;
        float* d_h_t = ssm->d_states + t * ssm->batch_size * ssm->state_dim;
        
        // Compute h_t = X_t @ B^T
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                ssm->state_dim,
                                ssm->batch_size,
                                ssm->input_dim,
                                &alpha,
                                ssm->d_B,
                                ssm->input_dim,
                                d_X_t,
                                ssm->input_dim,
                                &beta,
                                d_h_t,
                                ssm->state_dim));
        
        // Add h_t += h_{t-1} @ A^T if not first timestep
        if (t > 0) {
            float* d_h_prev = ssm->d_states + (t-1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    ssm->state_dim,
                                    ssm->batch_size,
                                    ssm->state_dim,
                                    &alpha,
                                    ssm->d_A,
                                    ssm->state_dim,
                                    d_h_prev,
                                    ssm->state_dim,
                                    &beta_add,
                                    d_h_t,
                                    ssm->state_dim));
        }
        
        // Apply Swish activation to state for output projection
        float* d_o_t = ssm->d_state_outputs + t * ssm->batch_size * ssm->state_dim;
        int block_size = 256;
        int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
        swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_o_t, d_h_t, ssm->batch_size * ssm->state_dim);
        
        // Compute outputs: y_t = swish(h_t) @ C^T + X_t @ D^T
        float* d_y_t = ssm->d_predictions + t * ssm->batch_size * ssm->output_dim;
        
        // y_t = swish(h_t) @ C^T
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                ssm->output_dim,
                                ssm->batch_size,
                                ssm->state_dim,
                                &alpha,
                                ssm->d_C,
                                ssm->state_dim,
                                d_o_t,
                                ssm->state_dim,
                                &beta,
                                d_y_t,
                                ssm->output_dim));
        
        // y_t += X_t @ D^T
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                ssm->output_dim,
                                ssm->batch_size,
                                ssm->input_dim,
                                &alpha,
                                ssm->d_D,
                                ssm->input_dim,
                                d_X_t,
                                ssm->input_dim,
                                &beta_add,
                                d_y_t,
                                ssm->output_dim));
    }
}

// Calculate loss
float calculate_loss_ssm(SSM* ssm, float* d_y) {
    int total_size = ssm->seq_len * ssm->batch_size * ssm->output_dim;
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;

    calc_error_kernel_ssm<<<num_blocks, block_size>>>(
        ssm->d_error,
        ssm->d_predictions,
        d_y,
        total_size
    );

    float loss;
    CHECK_CUBLAS(cublasSdot(ssm->cublas_handle, total_size, ssm->d_error, 1, ssm->d_error, 1, &loss));

    return loss / total_size;
}

// Zero gradients
void zero_gradients_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float)));
}

// Backward pass
void backward_pass_ssm(SSM* ssm, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Clear state errors
    CHECK_CUDA(cudaMemset(ssm->d_state_error, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
    
    for (int t = ssm->seq_len - 1; t >= 0; t--) {
        float* d_X_t = d_X + t * ssm->batch_size * ssm->input_dim;
        float* d_h_t = ssm->d_states + t * ssm->batch_size * ssm->state_dim;
        float* d_o_t = ssm->d_state_outputs + t * ssm->batch_size * ssm->state_dim;
        float* d_dy_t = ssm->d_error + t * ssm->batch_size * ssm->output_dim;
        float* d_dh_t = ssm->d_state_error + t * ssm->batch_size * ssm->state_dim;
        
        // Gradient w.r.t C: dC += dy_t^T @ swish(h_t)
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                ssm->state_dim,
                                ssm->output_dim,
                                ssm->batch_size,
                                &alpha,
                                d_o_t,
                                ssm->state_dim,
                                d_dy_t,
                                ssm->output_dim,
                                &beta_add,
                                ssm->d_C_grad,
                                ssm->state_dim));
        
        // Gradient w.r.t D: dD += dy_t^T @ X_t
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                ssm->input_dim,
                                ssm->output_dim,
                                ssm->batch_size,
                                &alpha,
                                d_X_t,
                                ssm->input_dim,
                                d_dy_t,
                                ssm->output_dim,
                                &beta_add,
                                ssm->d_D_grad,
                                ssm->input_dim));
        
        // Backprop through swish activation: do_t = dy_t @ C
        float* d_do_t = d_o_t; // reuse buffer
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                ssm->state_dim,
                                ssm->batch_size,
                                ssm->output_dim,
                                &alpha,
                                ssm->d_C,
                                ssm->state_dim,
                                d_dy_t,
                                ssm->output_dim,
                                &beta,
                                d_do_t,
                                ssm->state_dim));
        
        // Apply Swish derivative: dh_t = do_t * swish'(h_t)
        int block_size = 256;
        int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
        swish_backward_kernel_ssm<<<num_blocks, block_size>>>(d_dh_t, d_do_t, d_h_t, ssm->batch_size * ssm->state_dim);
        
        // Add gradient from next timestep if not last
        if (t < ssm->seq_len - 1) {
            float* d_dh_next = ssm->d_state_error + (t+1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    ssm->state_dim,
                                    ssm->batch_size,
                                    ssm->state_dim,
                                    &alpha,
                                    ssm->d_A,
                                    ssm->state_dim,
                                    d_dh_next,
                                    ssm->state_dim,
                                    &beta_add,
                                    d_dh_t,
                                    ssm->state_dim));
        }
        
        // Gradient w.r.t B: dB += dh_t^T @ X_t
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                ssm->input_dim,
                                ssm->state_dim,
                                ssm->batch_size,
                                &alpha,
                                d_X_t,
                                ssm->input_dim,
                                d_dh_t,
                                ssm->state_dim,
                                &beta_add,
                                ssm->d_B_grad,
                                ssm->input_dim));
        
        // Gradient w.r.t A: dA += dh_t^T @ h_{t-1}
        if (t > 0) {
            float* d_h_prev = ssm->d_states + (t-1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_T,
                                    ssm->state_dim,
                                    ssm->state_dim,
                                    ssm->batch_size,
                                    &alpha,
                                    d_h_prev,
                                    ssm->state_dim,
                                    d_dh_t,
                                    ssm->state_dim,
                                    &beta_add,
                                    ssm->d_A_grad,
                                    ssm->state_dim));
        }
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_ssm(
    float* weight,
    float* grad,
    float* m,
    float* v,
    float beta1,
    float beta2,
    float epsilon,
    float learning_rate,
    float weight_decay,
    float alpha_t,
    int size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update A
    int A_size = ssm->state_dim * ssm->state_dim;
    int A_blocks = (A_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<A_blocks, block_size>>>(
        ssm->d_A, ssm->d_A_grad, ssm->d_A_m, ssm->d_A_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, A_size, ssm->batch_size
    );
    
    // Update B
    int B_size = ssm->state_dim * ssm->input_dim;
    int B_blocks = (B_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<B_blocks, block_size>>>(
        ssm->d_B, ssm->d_B_grad, ssm->d_B_m, ssm->d_B_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, B_size, ssm->batch_size
    );
    
    // Update C
    int C_size = ssm->output_dim * ssm->state_dim;
    int C_blocks = (C_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<C_blocks, block_size>>>(
        ssm->d_C, ssm->d_C_grad, ssm->d_C_m, ssm->d_C_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, C_size, ssm->batch_size
    );
    
    // Update D
    int D_size = ssm->output_dim * ssm->input_dim;
    int D_blocks = (D_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<D_blocks, block_size>>>(
        ssm->d_D, ssm->d_D_grad, ssm->d_D_m, ssm->d_D_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, D_size, ssm->batch_size
    );
}

// Save model
void save_ssm(SSM* ssm, const char* filename) {
    // Allocate temporary host memory
    float* A = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    float* B = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* C = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(A, ssm->d_A, ssm->state_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B, ssm->d_B, ssm->state_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C, ssm->d_C, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D, ssm->d_D, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(A); free(B); free(C); free(D);
        return;
    }
    
    // Save dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->seq_len, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
    // Save matrices
    fwrite(A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Save Adam state
    float* A_m = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    float* A_v = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    float* B_m = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* B_v = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* C_m = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* C_v = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D_m = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    float* D_v = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(A_m, ssm->d_A_m, ssm->state_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(A_v, ssm->d_A_v, ssm->state_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_m, ssm->d_B_m, ssm->state_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_v, ssm->d_B_v, ssm->state_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_m, ssm->d_C_m, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_v, ssm->d_C_v, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_m, ssm->d_D_m, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_v, ssm->d_D_v, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(A_m, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(A_v, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(B_m, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(B_v, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Free temporary host memory
    free(A); free(B); free(C); free(D);
    free(A_m); free(A_v); free(B_m); free(B_v);
    free(C_m); free(C_v); free(D_m); free(D_v);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model
SSM* load_ssm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, state_dim, output_dim, seq_len, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Allocate temporary host memory
    float* A = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Load matrices
    fread(A, sizeof(float), state_dim * state_dim, file);
    fread(B, sizeof(float), state_dim * input_dim, file);
    fread(C, sizeof(float), output_dim * state_dim, file);
    fread(D, sizeof(float), output_dim * input_dim, file);
    
    fread(&ssm->t, sizeof(int), 1, file);
    
    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    float* A_m = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* A_v = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* B_m = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* B_v = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C_m = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* C_v = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D_m = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* D_v = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    fread(A_m, sizeof(float), state_dim * state_dim, file);
    fread(A_v, sizeof(float), state_dim * state_dim, file);
    fread(B_m, sizeof(float), state_dim * input_dim, file);
    fread(B_v, sizeof(float), state_dim * input_dim, file);
    fread(C_m, sizeof(float), output_dim * state_dim, file);
    fread(C_v, sizeof(float), output_dim * state_dim, file);
    fread(D_m, sizeof(float), output_dim * input_dim, file);
    fread(D_v, sizeof(float), output_dim * input_dim, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A_m, A_m, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_v, A_v, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_m, B_m, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_v, B_v, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_m, C_m, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_v, C_v, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m, D_m, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v, D_v, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(A); free(B); free(C); free(D);
    free(A_m); free(A_v); free(B_m); free(B_v);
    free(C_m); free(C_v); free(D_m); free(D_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}

#endif