#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// ---------------------------------------------------------------------
// Error checking macros for CUDA, cuBLAS and cuSOLVER calls
// ---------------------------------------------------------------------
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#ifndef CHECK_CUSOLVER
#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error in %s at line %d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// ---------------------------------------------------------------------
// Structure definition for the state-space model (SSM)
// This version uses cuBLAS with TF32 Tensor Cores and integrates the AdamW optimizer.
// ---------------------------------------------------------------------
typedef struct {
    // State transition matrices for layer 1 (stored on device)
    float* d_A1;  // state_dim x state_dim
    float* d_B1;  // state_dim x input_dim
    float* d_C1;  // output_dim x state_dim
    float* d_D1;  // output_dim x input_dim

    // State transition matrices for layer 2 (stored on device)
    float* d_A2;  // state_dim x state_dim
    float* d_B2;  // state_dim x input_dim
    float* d_C2;  // output_dim x state_dim
    float* d_D2;  // output_dim x input_dim

    // Host copies for saving/loading the model
    float* h_A1;
    float* h_B1;
    float* h_C1;
    float* h_D1;
    float* h_A2;
    float* h_B2;
    float* h_C2;
    float* h_D2;

    // Gradients (device pointers) - layer 1
    float* d_A1_grad;
    float* d_B1_grad;
    float* d_C1_grad;
    float* d_D1_grad;

    // Gradients (device pointers) - layer 2
    float* d_A2_grad;
    float* d_B2_grad;
    float* d_C2_grad;
    float* d_D2_grad;

    // Adam optimizer first (m) and second (v) moment estimates (device pointers) - layer 1
    float* d_A1_m;
    float* d_A1_v;
    float* d_B1_m;
    float* d_B1_v;
    float* d_C1_m;
    float* d_C1_v;
    float* d_D1_m;
    float* d_D1_v;

    // Adam optimizer first (m) and second (v) moment estimates (device pointers) - layer 2
    float* d_A2_m;
    float* d_A2_v;
    float* d_B2_m;
    float* d_B2_v;
    float* d_C2_m;
    float* d_C2_v;
    float* d_D2_m;
    float* d_D2_v;

    // Adam hyperparameters and counter
    float beta1;         // e.g., 0.9
    float beta2;         // e.g., 0.999
    float epsilon;       // e.g., 1e-8
    float weight_decay;  // e.g., 0.01
    int adam_t;          // time step counter

    // Helper arrays (device pointers) - layer 1
    float* d_state1;         // batch_size x state_dim
    float* d_next_state1;    // batch_size x state_dim
    float* d_pre_state1;     // batch_size x state_dim (pre-activation state)
    float* d_predictions1;   // batch_size x output_dim
    float* d_error1;         // batch_size x output_dim
    float* d_state_error1;   // batch_size x state_dim

    // Helper arrays (device pointers) - layer 2
    float* d_state2;         // batch_size x state_dim
    float* d_next_state2;    // batch_size x state_dim
    float* d_pre_state2;     // batch_size x state_dim (pre-activation state)
    float* d_predictions2;   // batch_size x output_dim
    float* d_error2;         // batch_size x output_dim
    float* d_state_error2;   // batch_size x state_dim

    // Residual connection buffer
    float* d_layer1_output;  // batch_size x output_dim

    // Final predictions after residual connection
    float* d_predictions;    // batch_size x output_dim

    // Temporary buffers for matrix operations
    float* d_temp_state;    // batch_size x state_dim
    float* d_temp_output;   // batch_size x output_dim
    
    // Spectral normalization buffers
    float* d_A_symm;        // state_dim x state_dim (for A^T*A)
    float* d_eigenvalues;   // state_dim
    float* d_work;          // work buffer for cuSOLVER
    int work_size;          // size of work buffer
    int* d_info;            // output info from cuSOLVER
    
    // CUDA library handles
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;

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
// CUDA kernel: Residual addition (element-wise)
// ---------------------------------------------------------------------
__global__ void residual_add_kernel(float* output, const float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += residual[idx];
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: AdamW update (per weight element)
// ---------------------------------------------------------------------
__global__ void adamw_update_kernel(float* W, const float* grad, float* m, float* v, 
                                      int size, float beta1, float beta2, float epsilon, 
                                      float weight_decay, float learning_rate, int batch_size, 
                                      float bias_correction1, float bias_correction2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / ((float) batch_size);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) - learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

// ---------------------------------------------------------------------
// Function: apply_spectral_normalization
// Normalizes matrix A to have spectral radius <= 0.999 using eigendecomposition
// ---------------------------------------------------------------------
void apply_spectral_normalization(SSM* ssm, float* d_A) {
    int n = ssm->state_dim;
    const float target_radius = 0.999f;
    const float alpha = 1.0f, beta = 0.0f;
    
    // Compute A^T * A (symmetric matrix with same singular values as A)
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              n, n, n, &alpha, d_A, CUDA_R_32F, n,
                              d_A, CUDA_R_32F, n, &beta,
                              ssm->d_A_symm, CUDA_R_32F, n,
                              CUBLAS_COMPUTE_32F_FAST_TF32,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    // Compute eigenvalues of A^T * A
    CHECK_CUSOLVER(cusolverDnSsyevd(
        ssm->cusolver_handle,
        CUSOLVER_EIG_MODE_NOVECTOR,  // Only compute eigenvalues
        CUBLAS_FILL_MODE_LOWER,      // Use lower triangular part
        n, ssm->d_A_symm, n, ssm->d_eigenvalues,
        ssm->d_work, ssm->work_size, ssm->d_info));
    
    // Check for error (optional since we pre-allocated everything)
    int info = 0;
    CHECK_CUDA(cudaMemcpy(&info, ssm->d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        fprintf(stderr, "Warning: Eigenvalue computation returned %d\n", info);
        return;
    }
    
    // Get largest eigenvalue (last one, they're sorted in ascending order)
    float max_eigenvalue;
    CHECK_CUDA(cudaMemcpy(&max_eigenvalue, &ssm->d_eigenvalues[n-1], 
                          sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute spectral radius (square root of largest eigenvalue of A^T * A)
    float spectral_radius = sqrtf(max_eigenvalue);
    
    // If spectral radius > 1.0, scale down matrix A
    if (spectral_radius > target_radius) {
        float scale_factor = target_radius / spectral_radius;
        CHECK_CUBLAS(cublasSscal(ssm->cublas_handle, n * n, &scale_factor, d_A, 1));
    }
}

// ---------------------------------------------------------------------
// Function: init_ssm
// Initializes the SSM structure, allocates host and device memory,
// sets initial weights with scaled random values, and copies them to device.
// Also initializes Adam optimizer parameters.
// ---------------------------------------------------------------------
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->batch_size = batch_size;

    // Set Adam hyperparameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->weight_decay = 0.01f;
    ssm->adam_t = 0;

    // Create cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&ssm->cublas_handle));
    
    // Enable TF32 tensor core math mode in cuBLAS
    CHECK_CUBLAS(cublasSetMathMode(ssm->cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    
    // Create cuSOLVER handle for spectral normalization
    CHECK_CUSOLVER(cusolverDnCreate(&ssm->cusolver_handle));

    // Allocate host memory for weight matrices - layer 1
    ssm->h_A1 = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->h_B1 = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->h_C1 = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->h_D1 = (float*)malloc(output_dim * input_dim * sizeof(float));

    // Allocate host memory for weight matrices - layer 2
    ssm->h_A2 = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->h_B2 = (float*)malloc(state_dim * output_dim * sizeof(float)); // Input to layer 2 is output_dim
    ssm->h_C2 = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->h_D2 = (float*)malloc(output_dim * output_dim * sizeof(float)); // D2 maps from layer 1 output to layer 2 output

    // Initialize matrices with scaled random values - layer 1
    float scale_A = 1.0f / sqrtf(state_dim);
    float scale_B = 1.0f / sqrtf(input_dim);
    float scale_C = 1.0f / sqrtf(state_dim);
    float scale_D = 1.0f / sqrtf(input_dim);

    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->h_A1[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_A;
    }
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->h_B1[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_B;
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->h_C1[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_C;
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->h_D1[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_D;
    }

    // Initialize matrices with scaled random values - layer 2
    float scale_A2 = 1.0f / sqrtf(state_dim);
    float scale_B2 = 1.0f / sqrtf(output_dim); // Input to layer 2 is output_dim
    float scale_C2 = 1.0f / sqrtf(state_dim);
    float scale_D2 = 1.0f / sqrtf(output_dim); // D2 maps from layer 1 output to layer 2 output

    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->h_A2[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_A2;
    }
    for (int i = 0; i < state_dim * output_dim; i++) {
        ssm->h_B2[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_B2;
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->h_C2[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_C2;
    }
    for (int i = 0; i < output_dim * output_dim; i++) {
        ssm->h_D2[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_D2;
    }

    // Allocate device memory for weight matrices - layer 1
    CHECK_CUDA(cudaMalloc(&ssm->d_A1, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B1, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C1, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D1, output_dim * input_dim * sizeof(float)));

    // Allocate device memory for weight matrices - layer 2
    CHECK_CUDA(cudaMalloc(&ssm->d_A2, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B2, state_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C2, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D2, output_dim * output_dim * sizeof(float)));

    // Allocate device memory for gradients - layer 1
    CHECK_CUDA(cudaMalloc(&ssm->d_A1_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B1_grad, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C1_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D1_grad, output_dim * input_dim * sizeof(float)));

    // Allocate device memory for gradients - layer 2
    CHECK_CUDA(cudaMalloc(&ssm->d_A2_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B2_grad, state_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C2_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D2_grad, output_dim * output_dim * sizeof(float)));

    // Allocate device memory for Adam first and second moment estimates and initialize to zero - layer 1
    CHECK_CUDA(cudaMalloc(&ssm->d_A1_m, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A1_v, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B1_m, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B1_v, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C1_m, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C1_v, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D1_m, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D1_v, output_dim * input_dim * sizeof(float)));

    // Allocate device memory for Adam first and second moment estimates and initialize to zero - layer 2
    CHECK_CUDA(cudaMalloc(&ssm->d_A2_m, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A2_v, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B2_m, state_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B2_v, state_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C2_m, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C2_v, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D2_m, output_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D2_v, output_dim * output_dim * sizeof(float)));

    CHECK_CUDA(cudaMemset(ssm->d_A1_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A1_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B1_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B1_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C1_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C1_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D1_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D1_v, 0, output_dim * input_dim * sizeof(float)));

    CHECK_CUDA(cudaMemset(ssm->d_A2_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A2_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B2_m, 0, state_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B2_v, 0, state_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C2_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C2_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D2_m, 0, output_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D2_v, 0, output_dim * output_dim * sizeof(float)));

    // Allocate helper arrays - layer 1
    CHECK_CUDA(cudaMalloc(&ssm->d_state1, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_next_state1, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_pre_state1, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions1, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error1, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error1, batch_size * state_dim * sizeof(float)));

    // Allocate helper arrays - layer 2
    CHECK_CUDA(cudaMalloc(&ssm->d_state2, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_next_state2, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_pre_state2, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions2, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error2, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error2, batch_size * state_dim * sizeof(float)));

    // Residual connection buffer
    CHECK_CUDA(cudaMalloc(&ssm->d_layer1_output, batch_size * output_dim * sizeof(float)));

    // Final predictions
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions, batch_size * output_dim * sizeof(float)));

    // Allocate temporary buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_output, batch_size * output_dim * sizeof(float)));
    
    // Allocate spectral normalization resources
    CHECK_CUDA(cudaMalloc(&ssm->d_A_symm, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_eigenvalues, state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_info, sizeof(int)));
    
    // Get workspace size for eigenvalue computation
    CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(
        ssm->cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
        state_dim, ssm->d_A_symm, state_dim, ssm->d_eigenvalues, &ssm->work_size));
    
    // Allocate workspace
    CHECK_CUDA(cudaMalloc(&ssm->d_work, ssm->work_size * sizeof(float)));

    // Copy weight matrices from host to device - layer 1
    CHECK_CUDA(cudaMemcpy(ssm->d_A1, ssm->h_A1, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B1, ssm->h_B1, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C1, ssm->h_C1, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D1, ssm->h_D1, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Copy weight matrices from host to device - layer 2
    CHECK_CUDA(cudaMemcpy(ssm->d_A2, ssm->h_A2, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B2, ssm->h_B2, state_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C2, ssm->h_C2, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D2, ssm->h_D2, output_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize states to zero
    CHECK_CUDA(cudaMemset(ssm->d_state1, 0, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_state2, 0, batch_size * state_dim * sizeof(float)));
    
    // Apply spectral normalization to initial A matrices
    apply_spectral_normalization(ssm, ssm->d_A1);
    apply_spectral_normalization(ssm, ssm->d_A2);

    return ssm;
}

// ---------------------------------------------------------------------
// Function: layer_forward_pass
// Computes the forward pass for one layer:
//   pre_state = A * state + B * input
//   next_state = swish(pre_state)
//   predictions = C * next_state + D * input
// Updates the internal state to next_state.
// ---------------------------------------------------------------------
void layer_forward_pass(SSM* ssm, float* d_input, float* d_A, float* d_B, 
                        float* d_C, float* d_D, float* d_state, 
                        float* d_pre_state, float* d_next_state, 
                        float* d_predictions, int input_dim) {
    const float alpha = 1.0f, beta = 0.0f;

    // Compute pre_state = A * state
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->state_dim, ssm->batch_size, ssm->state_dim,
                             &alpha,
                             d_A, CUDA_R_32F, ssm->state_dim,
                             d_state, CUDA_R_32F, ssm->state_dim,
                             &beta,
                             d_pre_state, CUDA_R_32F, ssm->state_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Add input contribution: pre_state += B * input
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->state_dim, ssm->batch_size, input_dim,
                             &alpha,
                             d_B, CUDA_R_32F, ssm->state_dim,
                             d_input, CUDA_R_32F, input_dim,
                             &alpha, // Add to existing pre_state
                             d_pre_state, CUDA_R_32F, ssm->state_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Apply swish activation: next_state = swish(pre_state)
    int total_state = ssm->batch_size * ssm->state_dim;
    int block_size = 256;
    int num_blocks = (total_state + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(d_next_state, d_pre_state, total_state);
    
    // Compute predictions = C * next_state
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->output_dim, ssm->batch_size, ssm->state_dim,
                             &alpha,
                             d_C, CUDA_R_32F, ssm->output_dim,
                             d_next_state, CUDA_R_32F, ssm->state_dim,
                             &beta,
                             d_predictions, CUDA_R_32F, ssm->output_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                             
    // Add direct feedthrough: predictions += D * input
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             ssm->output_dim, ssm->batch_size, input_dim,
                             &alpha,
                             d_D, CUDA_R_32F, ssm->output_dim,
                             d_input, CUDA_R_32F, input_dim,
                             &alpha, // Add to existing predictions
                             d_predictions, CUDA_R_32F, ssm->output_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    // Update internal state: state = next_state
    CHECK_CUDA(cudaMemcpy(d_state, d_next_state,
                          ssm->batch_size * ssm->state_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------
// Function: forward_pass
// Computes the forward pass through both layers with a residual connection:
//   - First layer processes input X
//   - Second layer processes output of first layer
//   - Final output adds layer 1 output (residual connection)
// ---------------------------------------------------------------------
void forward_pass(SSM* ssm, float* d_X) {
    // Save copy of layer 1 input for the backward pass
    CHECK_CUDA(cudaMemcpy(ssm->d_temp_state, d_X, 
                          ssm->batch_size * ssm->input_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
                          
    // Layer 1 forward pass
    layer_forward_pass(
        ssm, d_X, 
        ssm->d_A1, ssm->d_B1, ssm->d_C1, ssm->d_D1,
        ssm->d_state1, ssm->d_pre_state1, ssm->d_next_state1, 
        ssm->d_predictions1, ssm->input_dim
    );
    
    // Save layer 1 output for residual connection
    CHECK_CUDA(cudaMemcpy(ssm->d_layer1_output, ssm->d_predictions1,
                          ssm->batch_size * ssm->output_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // Layer 2 forward pass (using layer 1 output as input)
    layer_forward_pass(
        ssm, ssm->d_predictions1,
        ssm->d_A2, ssm->d_B2, ssm->d_C2, ssm->d_D2,
        ssm->d_state2, ssm->d_pre_state2, ssm->d_next_state2,
        ssm->d_predictions2, ssm->output_dim
    );
    
    // Copy layer 2 output to final predictions
    CHECK_CUDA(cudaMemcpy(ssm->d_predictions, ssm->d_predictions2,
                          ssm->batch_size * ssm->output_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // Add residual connection: predictions += layer1_output
    int total_outputs = ssm->batch_size * ssm->output_dim;
    int block_size = 256;
    int num_blocks = (total_outputs + block_size - 1) / block_size;
    residual_add_kernel<<<num_blocks, block_size>>>(ssm->d_predictions, ssm->d_layer1_output, total_outputs);
}

// ---------------------------------------------------------------------
// Function: calculate_loss
// Computes the Mean Squared Error loss between predictions and targets.
// ---------------------------------------------------------------------
float calculate_loss(SSM* ssm, float* d_y) {
    int size = ssm->batch_size * ssm->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    mse_loss_kernel<<<num_blocks, block_size>>>(ssm->d_error2, ssm->d_predictions, d_y, size);
    float loss = 0.0f;
    CHECK_CUBLAS(cublasSdot(ssm->cublas_handle, size,
                            ssm->d_error2, 1,
                            ssm->d_error2, 1,
                            &loss));
    return loss / size;
}

// ---------------------------------------------------------------------
// Function: zero_gradients
// Clears the gradient arrays on the device.
// ---------------------------------------------------------------------
void zero_gradients(SSM* ssm) {
    int size_A = ssm->state_dim * ssm->state_dim * sizeof(float);
    int size_B1 = ssm->state_dim * ssm->input_dim * sizeof(float);
    int size_B2 = ssm->state_dim * ssm->output_dim * sizeof(float);
    int size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    int size_D1 = ssm->output_dim * ssm->input_dim * sizeof(float);
    int size_D2 = ssm->output_dim * ssm->output_dim * sizeof(float);
    
    // Layer 1
    CHECK_CUDA(cudaMemset(ssm->d_A1_grad, 0, size_A));
    CHECK_CUDA(cudaMemset(ssm->d_B1_grad, 0, size_B1));
    CHECK_CUDA(cudaMemset(ssm->d_C1_grad, 0, size_C));
    CHECK_CUDA(cudaMemset(ssm->d_D1_grad, 0, size_D1));
    
    // Layer 2
    CHECK_CUDA(cudaMemset(ssm->d_A2_grad, 0, size_A));
    CHECK_CUDA(cudaMemset(ssm->d_B2_grad, 0, size_B2));
    CHECK_CUDA(cudaMemset(ssm->d_C2_grad, 0, size_C));
    CHECK_CUDA(cudaMemset(ssm->d_D2_grad, 0, size_D2));
}

// ---------------------------------------------------------------------
// Function: layer_backward_pass
// Computes gradients for a single layer given error
// ---------------------------------------------------------------------
void layer_backward_pass(SSM* ssm, float* d_input, float* d_error,
                        float* d_C,
                        float* d_state, float* d_next_state, float* d_pre_state,
                        float* d_state_error, float* d_A_grad, float* d_B_grad,
                        float* d_C_grad, float* d_D_grad, int input_dim) {
    const float alpha = 1.0f, beta = 0.0f;

    // Gradient for C: d_C_grad = error * (next_state)^T
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->output_dim, ssm->state_dim, ssm->batch_size,
                             &alpha,
                             d_error, CUDA_R_32F, ssm->output_dim,
                             d_next_state, CUDA_R_32F, ssm->state_dim,
                             &beta,
                             d_C_grad, CUDA_R_32F, ssm->output_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                             
    // Gradient for D: d_D_grad = error * (input)^T
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->output_dim, input_dim, ssm->batch_size,
                             &alpha,
                             d_error, CUDA_R_32F, ssm->output_dim,
                             d_input, CUDA_R_32F, input_dim,
                             &beta,
                             d_D_grad, CUDA_R_32F, ssm->output_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                             
    // Compute state error: state_error = C^T * error
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             ssm->state_dim, ssm->batch_size, ssm->output_dim,
                             &alpha,
                             d_C, CUDA_R_32F, ssm->output_dim,
                             d_error, CUDA_R_32F, ssm->output_dim,
                             &beta,
                             d_state_error, CUDA_R_32F, ssm->state_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                             
    // Apply swish backward: modify state_error in place
    int total_state = ssm->batch_size * ssm->state_dim;
    int block_size = 256;
    int num_blocks = (total_state + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(d_state_error, 
                                                      d_pre_state, 
                                                      d_next_state, 
                                                      total_state);
                                                      
    // Gradient for A: d_A_grad = state_error * (state)^T
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->state_dim, ssm->state_dim, ssm->batch_size,
                             &alpha,
                             d_state_error, CUDA_R_32F, ssm->state_dim,
                             d_state, CUDA_R_32F, ssm->state_dim,
                             &beta,
                             d_A_grad, CUDA_R_32F, ssm->state_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                             
    // Gradient for B: d_B_grad = state_error * (input)^T
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             ssm->state_dim, input_dim, ssm->batch_size,
                             &alpha,
                             d_state_error, CUDA_R_32F, ssm->state_dim,
                             d_input, CUDA_R_32F, input_dim,
                             &beta,
                             d_B_grad, CUDA_R_32F, ssm->state_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------------
// Function: backward_pass
// Computes gradients through both layers using the chain rule with 
// residual connection:
//   - First backprop through layer 2
//   - Propagate error to layer 1 (adding residual gradient)
//   - Backprop through layer 1
// ---------------------------------------------------------------------
void backward_pass(SSM* ssm, float* d_X) {
    const float alpha = 1.0f, beta_add = 1.0f;

    // Copy error to layer 2 error
    CHECK_CUDA(cudaMemcpy(ssm->d_error2, ssm->d_error2, 
                          ssm->batch_size * ssm->output_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Backward pass for layer 2
    layer_backward_pass(
        ssm, ssm->d_layer1_output, ssm->d_error2,
        ssm->d_C2, 
        ssm->d_state2, ssm->d_next_state2, ssm->d_pre_state2,
        ssm->d_state_error2, ssm->d_A2_grad, ssm->d_B2_grad,
        ssm->d_C2_grad, ssm->d_D2_grad, ssm->output_dim
    );

    // Compute error for layer 1 (from layer 2 and residual)
    // Layer 1 error = layer 2 residual error + B2^T * state_error2
    
    // Copy the residual error to layer 1 error (start with residual gradient)
    CHECK_CUDA(cudaMemcpy(ssm->d_error1, ssm->d_error2,
                          ssm->batch_size * ssm->output_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Add error from layer 2's B matrix: error1 += B2^T * state_error2
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             ssm->output_dim, ssm->batch_size, ssm->state_dim,
                             &alpha,
                             ssm->d_B2, CUDA_R_32F, ssm->state_dim,
                             ssm->d_state_error2, CUDA_R_32F, ssm->state_dim,
                             &beta_add, // Add to existing error
                             ssm->d_error1, CUDA_R_32F, ssm->output_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                             
    // Add error from layer 2's D matrix: error1 += D2^T * error2
    CHECK_CUBLAS(cublasGemmEx(ssm->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             ssm->output_dim, ssm->batch_size, ssm->output_dim,
                             &alpha,
                             ssm->d_D2, CUDA_R_32F, ssm->output_dim,
                             ssm->d_error2, CUDA_R_32F, ssm->output_dim,
                             &beta_add, // Add to existing error
                             ssm->d_error1, CUDA_R_32F, ssm->output_dim,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Backward pass for layer 1
    layer_backward_pass(
        ssm, d_X, ssm->d_error1,
        ssm->d_C1,
        ssm->d_state1, ssm->d_next_state1, ssm->d_pre_state1,
        ssm->d_state_error1, ssm->d_A1_grad, ssm->d_B1_grad,
        ssm->d_C1_grad, ssm->d_D1_grad, ssm->input_dim
    );
}

// ---------------------------------------------------------------------
// Function: update_weights
// Uses the AdamW optimizer to update each weight matrix and applies
// spectral normalization to A periodically to ensure stability.
// ---------------------------------------------------------------------
void update_weights(SSM* ssm, float learning_rate) {
    ssm->adam_t++; // Increment time step
    float bias_correction1 = 1.0f - powf(ssm->beta1, (float)ssm->adam_t);
    float bias_correction2 = 1.0f - powf(ssm->beta2, (float)ssm->adam_t);

    int block_size = 256;
    int size_A = ssm->state_dim * ssm->state_dim;
    int size_B1 = ssm->state_dim * ssm->input_dim;
    int size_B2 = ssm->state_dim * ssm->output_dim;
    int size_C = ssm->output_dim * ssm->state_dim;
    int size_D1 = ssm->output_dim * ssm->input_dim;
    int size_D2 = ssm->output_dim * ssm->output_dim;

    int num_blocks_A = (size_A + block_size - 1) / block_size;
    int num_blocks_B1 = (size_B1 + block_size - 1) / block_size;
    int num_blocks_B2 = (size_B2 + block_size - 1) / block_size;
    int num_blocks_C = (size_C + block_size - 1) / block_size;
    int num_blocks_D1 = (size_D1 + block_size - 1) / block_size;
    int num_blocks_D2 = (size_D2 + block_size - 1) / block_size;

    // Update layer 1 weights
    adamw_update_kernel<<<num_blocks_A, block_size>>>(
        ssm->d_A1, ssm->d_A1_grad, ssm->d_A1_m, ssm->d_A1_v,
        size_A, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_B1, block_size>>>(
        ssm->d_B1, ssm->d_B1_grad, ssm->d_B1_m, ssm->d_B1_v,
        size_B1, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_C, block_size>>>(
        ssm->d_C1, ssm->d_C1_grad, ssm->d_C1_m, ssm->d_C1_v,
        size_C, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_D1, block_size>>>(
        ssm->d_D1, ssm->d_D1_grad, ssm->d_D1_m, ssm->d_D1_v,
        size_D1, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    // Update layer 2 weights
    adamw_update_kernel<<<num_blocks_A, block_size>>>(
        ssm->d_A2, ssm->d_A2_grad, ssm->d_A2_m, ssm->d_A2_v,
        size_A, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_B2, block_size>>>(
        ssm->d_B2, ssm->d_B2_grad, ssm->d_B2_m, ssm->d_B2_v,
        size_B2, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_C, block_size>>>(
        ssm->d_C2, ssm->d_C2_grad, ssm->d_C2_m, ssm->d_C2_v,
        size_C, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_D2, block_size>>>(
        ssm->d_D2, ssm->d_D2_grad, ssm->d_D2_m, ssm->d_D2_v,
        size_D2, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);
        
    // Apply spectral normalization to A matrices periodically
    if (ssm->adam_t % 10 == 0) {
        apply_spectral_normalization(ssm, ssm->d_A1);
        apply_spectral_normalization(ssm, ssm->d_A2);
    }
}

// ---------------------------------------------------------------------
// Function: free_ssm
// Frees all allocated memory (both device and host) and destroys the cuBLAS handle.
// ---------------------------------------------------------------------
void free_ssm(SSM* ssm) {
    // Free device memory - layer 1
    cudaFree(ssm->d_A1);
    cudaFree(ssm->d_B1);
    cudaFree(ssm->d_C1);
    cudaFree(ssm->d_D1);
    cudaFree(ssm->d_A1_grad);
    cudaFree(ssm->d_B1_grad);
    cudaFree(ssm->d_C1_grad);
    cudaFree(ssm->d_D1_grad);
    cudaFree(ssm->d_A1_m);
    cudaFree(ssm->d_A1_v);
    cudaFree(ssm->d_B1_m);
    cudaFree(ssm->d_B1_v);
    cudaFree(ssm->d_C1_m);
    cudaFree(ssm->d_C1_v);
    cudaFree(ssm->d_D1_m);
    cudaFree(ssm->d_D1_v);
    cudaFree(ssm->d_state1);
    cudaFree(ssm->d_next_state1);
    cudaFree(ssm->d_pre_state1);
    cudaFree(ssm->d_predictions1);
    cudaFree(ssm->d_error1);
    cudaFree(ssm->d_state_error1);
    
    // Free device memory - layer 2
    cudaFree(ssm->d_A2);
    cudaFree(ssm->d_B2);
    cudaFree(ssm->d_C2);
    cudaFree(ssm->d_D2);
    cudaFree(ssm->d_A2_grad);
    cudaFree(ssm->d_B2_grad);
    cudaFree(ssm->d_C2_grad);
    cudaFree(ssm->d_D2_grad);
    cudaFree(ssm->d_A2_m);
    cudaFree(ssm->d_A2_v);
    cudaFree(ssm->d_B2_m);
    cudaFree(ssm->d_B2_v);
    cudaFree(ssm->d_C2_m);
    cudaFree(ssm->d_C2_v);
    cudaFree(ssm->d_D2_m);
    cudaFree(ssm->d_D2_v);
    cudaFree(ssm->d_state2);
    cudaFree(ssm->d_next_state2);
    cudaFree(ssm->d_pre_state2);
    cudaFree(ssm->d_predictions2);
    cudaFree(ssm->d_error2);
    cudaFree(ssm->d_state_error2);
    
    // Free residual connection and final prediction buffers
    cudaFree(ssm->d_layer1_output);
    cudaFree(ssm->d_predictions);
    
    // Free temporary buffers
    cudaFree(ssm->d_temp_state);
    cudaFree(ssm->d_temp_output);
    
    // Free spectral normalization resources
    cudaFree(ssm->d_A_symm);
    cudaFree(ssm->d_eigenvalues);
    cudaFree(ssm->d_work);
    cudaFree(ssm->d_info);

    // Free host memory
    free(ssm->h_A1);
    free(ssm->h_B1);
    free(ssm->h_C1);
    free(ssm->h_D1);
    free(ssm->h_A2);
    free(ssm->h_B2);
    free(ssm->h_C2);
    free(ssm->h_D2);

    // Destroy handles
    cublasDestroy(ssm->cublas_handle);
    cusolverDnDestroy(ssm->cusolver_handle);

    free(ssm);
}

// ---------------------------------------------------------------------
// Function: save_model
// Saves the model weights to a binary file.
// ---------------------------------------------------------------------
void save_ssm(SSM* ssm, const char* filename) {
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
    size_t size_B1 = ssm->state_dim * ssm->input_dim * sizeof(float);
    size_t size_B2 = ssm->state_dim * ssm->output_dim * sizeof(float);
    size_t size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    size_t size_D1 = ssm->output_dim * ssm->input_dim * sizeof(float);
    size_t size_D2 = ssm->output_dim * ssm->output_dim * sizeof(float);

    // Allocate temporary host buffers
    float* h_A1 = (float*)malloc(size_A);
    float* h_B1 = (float*)malloc(size_B1);
    float* h_C1 = (float*)malloc(size_C);
    float* h_D1 = (float*)malloc(size_D1);
    float* h_A2 = (float*)malloc(size_A);
    float* h_B2 = (float*)malloc(size_B2);
    float* h_C2 = (float*)malloc(size_C);
    float* h_D2 = (float*)malloc(size_D2);

    // Copy weight matrices from device to host
    CHECK_CUDA(cudaMemcpy(h_A1, ssm->d_A1, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B1, ssm->d_B1, size_B1, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C1, ssm->d_C1, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D1, ssm->d_D1, size_D1, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_A2, ssm->d_A2, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B2, ssm->d_B2, size_B2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C2, ssm->d_C2, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D2, ssm->d_D2, size_D2, cudaMemcpyDeviceToHost));

    // Write weight matrices to file
    fwrite(h_A1, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B1, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_C1, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D1, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(h_A2, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B2, sizeof(float), ssm->state_dim * ssm->output_dim, file);
    fwrite(h_C2, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D2, sizeof(float), ssm->output_dim * ssm->output_dim, file);

    free(h_A1);
    free(h_B1);
    free(h_C1);
    free(h_D1);
    free(h_A2);
    free(h_B2);
    free(h_C2);
    free(h_D2);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// ---------------------------------------------------------------------
// Function: load_model
// Loads the model weights from a binary file and initializes a new SSM.
// ---------------------------------------------------------------------
SSM* load_ssm(const char* filename) {
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
    size_t size_B1 = state_dim * input_dim * sizeof(float);
    size_t size_B2 = state_dim * output_dim * sizeof(float);
    size_t size_C = output_dim * state_dim * sizeof(float);
    size_t size_D1 = output_dim * input_dim * sizeof(float);
    size_t size_D2 = output_dim * output_dim * sizeof(float);

    float* h_A1 = (float*)malloc(size_A);
    float* h_B1 = (float*)malloc(size_B1);
    float* h_C1 = (float*)malloc(size_C);
    float* h_D1 = (float*)malloc(size_D1);
    float* h_A2 = (float*)malloc(size_A);
    float* h_B2 = (float*)malloc(size_B2);
    float* h_C2 = (float*)malloc(size_C);
    float* h_D2 = (float*)malloc(size_D2);

    fread(h_A1, sizeof(float), state_dim * state_dim, file);
    fread(h_B1, sizeof(float), state_dim * input_dim, file);
    fread(h_C1, sizeof(float), output_dim * state_dim, file);
    fread(h_D1, sizeof(float), output_dim * input_dim, file);
    fread(h_A2, sizeof(float), state_dim * state_dim, file);
    fread(h_B2, sizeof(float), state_dim * output_dim, file);
    fread(h_C2, sizeof(float), output_dim * state_dim, file);
    fread(h_D2, sizeof(float), output_dim * output_dim, file);

    CHECK_CUDA(cudaMemcpy(ssm->d_A1, h_A1, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B1, h_B1, size_B1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C1, h_C1, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D1, h_D1, size_D1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A2, h_A2, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B2, h_B2, size_B2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C2, h_C2, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D2, h_D2, size_D2, cudaMemcpyHostToDevice));

    free(h_A1);
    free(h_B1);
    free(h_C1);
    free(h_D1);
    free(h_A2);
    free(h_B2);
    free(h_C2);
    free(h_D2);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    return ssm;
}

#endif