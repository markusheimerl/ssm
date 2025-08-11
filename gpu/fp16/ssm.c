#include "ssm.h"

// Initialize the network with configurable dimensions
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size, cublasHandle_t cublas_handle) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    
    // Store dimensions
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->seq_len = seq_len;
    ssm->batch_size = batch_size;
    
    // Initialize Adam parameters
    ssm->beta1 = __float2half(0.9f);
    ssm->beta2 = __float2half(0.999f);
    ssm->epsilon = __float2half(1e-4f);
    ssm->t = 0;
    ssm->weight_decay = __float2half(0.01f);
    
    // Initialize cuBLAS
    ssm->cublas_handle = cublas_handle;
    
    // Allocate host memory for weights
    __half* A = (__half*)malloc(state_dim * state_dim * sizeof(__half));
    __half* B = (__half*)malloc(state_dim * input_dim * sizeof(__half));
    __half* C = (__half*)malloc(output_dim * state_dim * sizeof(__half));
    __half* D = (__half*)malloc(output_dim * input_dim * sizeof(__half));
    
    // Initialize weights on host
    __half scale_A = __float2half(0.5f / sqrtf(state_dim));
    __half scale_B = __float2half(1.5f / sqrtf(input_dim));
    __half scale_C = __float2half(1.5f / sqrtf(state_dim));
    __half scale_D = __float2half(1.5f / sqrtf(input_dim));
    
    for (int i = 0; i < state_dim * state_dim; i++) {
        A[i] = __hmul(__float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f)), scale_A);
    }
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        B[i] = __hmul(__float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f)), scale_B);
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        C[i] = __hmul(__float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f)), scale_C);
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        D[i] = __hmul(__float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f)), scale_D);
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A, state_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B, state_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, state_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad, state_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, output_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, output_dim * input_dim * sizeof(__half)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ssm->d_A_m, state_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_v, state_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_m, state_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_v, state_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_m, output_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_v, output_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_m, output_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_v, output_dim * input_dim * sizeof(__half)));
    
    // Allocate device memory for layer outputs and working buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_layer1_preact, seq_len * batch_size * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_layer1_output, seq_len * batch_size * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_layer2_output, seq_len * batch_size * output_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error_hidden, seq_len * batch_size * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error_output, seq_len * batch_size * output_dim * sizeof(__half)));
    
    // Allocate device memory for loss computation
    CHECK_CUDA(cudaMalloc(&ssm->d_loss, sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, state_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, state_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, state_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, state_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(__half)));
    
    // Free local host memory
    free(A); free(B); free(C); free(D);
    
    return ssm;
}

// Free network memory
void free_ssm(SSM* ssm) {
    // Free device memory
    cudaFree(ssm->d_A); cudaFree(ssm->d_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);
    cudaFree(ssm->d_A_grad); cudaFree(ssm->d_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_A_m); cudaFree(ssm->d_A_v);
    cudaFree(ssm->d_B_m); cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m); cudaFree(ssm->d_C_v);
    cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);
    cudaFree(ssm->d_layer1_preact); cudaFree(ssm->d_layer1_output); cudaFree(ssm->d_layer2_output);
    cudaFree(ssm->d_error_output); cudaFree(ssm->d_error_hidden);
    cudaFree(ssm->d_loss);
    free(ssm);
}

// Reset state for new sequence
void reset_state_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_layer1_preact, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_layer1_output, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_layer2_output, 0, ssm->seq_len * ssm->batch_size * ssm->output_dim * sizeof(__half)));
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel_ssm(__half* output, __half* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __half h = input[idx];
        __half one = __float2half(1.0f);
        __half neg_h = __hneg(h);
        __half exp_neg_h = hexp(neg_h);
        __half denom = __hadd(one, exp_neg_h);
        output[idx] = __hdiv(h, denom);
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel_ssm(__half* grad_input, __half* grad_output, __half* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __half h = input[idx];
        __half one = __float2half(1.0f);
        __half neg_h = __hneg(h);
        __half exp_neg_h = hexp(neg_h);
        __half sigmoid = __hdiv(one, __hadd(one, exp_neg_h));
        __half one_minus_sigmoid = __hsub(one, sigmoid);
        __half h_sigmoid = __hmul(h, sigmoid);
        __half derivative = __hadd(sigmoid, __hmul(h_sigmoid, one_minus_sigmoid));
        grad_input[idx] = __hmul(grad_output[idx], derivative);
    }
}

// CUDA kernel for FP16 matrix addition/subtraction
__global__ void hgeam_kernel_ssm(__half* C, __half alpha, __half* A, __half beta, __half* B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = rows * cols;
    
    if (idx < total_size) {
        __half a_val = __hmul(alpha, A[idx]);
        __half b_val = __hmul(beta, B[idx]);
        C[idx] = __hadd(a_val, b_val);
    }
}

// CUDA kernel for FP16 dot product
__global__ void hdot_kernel_ssm(__half* x, float* result, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory and compute partial sum
    float temp = 0.0f;
    if (idx < size) {
        __half val = x[idx];
        float val_f = __half2float(val);
        temp = val_f * val_f;
    }
    sdata[tid] = temp;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Forward pass for single timestep
void forward_pass_ssm(SSM* ssm, __half* d_X_t, int timestep) {
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    // Get pointers to current timestep data (time-major format)
    __half* d_H_t = &ssm->d_layer1_preact[timestep * ssm->batch_size * ssm->state_dim];
    __half* d_S_t = &ssm->d_layer1_output[timestep * ssm->batch_size * ssm->state_dim];
    __half* d_Y_t = &ssm->d_layer2_output[timestep * ssm->batch_size * ssm->output_dim];
    
    // H_t = X_t B^T
    CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_B, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &beta, d_H_t, ssm->state_dim));
    
    // H_t = H_t + H_{t-1} A^T
    if (timestep > 0) {
        __half* d_H_prev = &ssm->d_layer1_preact[(timestep-1) * ssm->batch_size * ssm->state_dim];
        CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                &alpha, ssm->d_A, ssm->state_dim,
                                d_H_prev, ssm->state_dim,
                                &alpha, d_H_t, ssm->state_dim));
    }
    
    // S_t = H_tσ(H_t)
    int block_size = 256;
    int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_S_t, d_H_t, ssm->batch_size * ssm->state_dim);
    
    // Y_t = S_t C^T
    CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->state_dim,
                            &alpha, ssm->d_C, ssm->state_dim,
                            d_S_t, ssm->state_dim,
                            &beta, d_Y_t, ssm->output_dim));
    
    // Y_t = Y_t + X_t D^T
    CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_D, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &alpha, d_Y_t, ssm->output_dim));
}

// Calculate loss
float calculate_loss_ssm(SSM* ssm, __half* d_y) {
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(-1.0f);
    int total_size = ssm->seq_len * ssm->batch_size * ssm->output_dim;
    
    // Reset loss to zero
    CHECK_CUDA(cudaMemset(ssm->d_loss, 0, sizeof(float)));
    
    // ∂L/∂Y = Y - Y_true
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    hgeam_kernel_ssm<<<num_blocks, block_size>>>(
        ssm->d_error_output, alpha, ssm->d_layer2_output, beta, d_y, 
        ssm->output_dim, ssm->seq_len * ssm->batch_size);
    
    // Compute dot product
    int shared_mem_size = block_size * sizeof(float);
    hdot_kernel_ssm<<<num_blocks, block_size, shared_mem_size>>>(
        ssm->d_error_output, ssm->d_loss, total_size);
    
    // Copy result back to host
    float loss;
    CHECK_CUDA(cudaMemcpy(&loss, ssm->d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    return loss / total_size;
}

// Zero gradients
void zero_gradients_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(__half)));
}

// Backward pass for single timestep
void backward_pass_ssm(SSM* ssm, __half* d_X_t, int timestep) {
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    // Get pointers to current timestep data
    __half* d_H_t = &ssm->d_layer1_preact[timestep * ssm->batch_size * ssm->state_dim];
    __half* d_S_t = &ssm->d_layer1_output[timestep * ssm->batch_size * ssm->state_dim];
    __half* d_error_output_t = &ssm->d_error_output[timestep * ssm->batch_size * ssm->output_dim];
    __half* d_error_hidden_t = &ssm->d_error_hidden[timestep * ssm->batch_size * ssm->state_dim];
    
    // ∂L/∂C += (∂L/∂Y_t)^T S_t
    CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->state_dim, ssm->output_dim, ssm->batch_size,
                            &alpha, d_S_t, ssm->state_dim,
                            d_error_output_t, ssm->output_dim,
                            &alpha, ssm->d_C_grad, ssm->state_dim));
    
    // ∂L/∂D += (∂L/∂Y_t)^T X_t
    CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->input_dim, ssm->output_dim, ssm->batch_size,
                            &alpha, d_X_t, ssm->input_dim,
                            d_error_output_t, ssm->output_dim,
                            &alpha, ssm->d_D_grad, ssm->input_dim));
    
    // ∂L/∂S_t = (∂L/∂Y_t) C
    CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->output_dim,
                            &alpha, ssm->d_C, ssm->state_dim,
                            d_error_output_t, ssm->output_dim,
                            &beta, d_error_hidden_t, ssm->state_dim));
    
    // ∂L/∂H_t = ∂L/∂S_t ⊙ [σ(H_t) + H_t σ(H_t)(1-σ(H_t))]
    int block_size = 256;
    int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    swish_backward_kernel_ssm<<<num_blocks, block_size>>>(d_error_hidden_t, d_error_hidden_t, d_H_t, ssm->batch_size * ssm->state_dim);
    
    // ∂L/∂B += (∂L/∂H_t)^T X_t
    CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->input_dim, ssm->state_dim, ssm->batch_size,
                            &alpha, d_X_t, ssm->input_dim,
                            d_error_hidden_t, ssm->state_dim,
                            &alpha, ssm->d_B_grad, ssm->input_dim));
    
    // Propagate error to previous timestep
    if (timestep > 0) {
        __half* d_H_prev = &ssm->d_layer1_preact[(timestep-1) * ssm->batch_size * ssm->state_dim];
        __half* d_error_hidden_prev = &ssm->d_error_hidden[(timestep-1) * ssm->batch_size * ssm->state_dim];
        
        // ∂L/∂A += (∂L/∂H_t)^T H_{t-1}
        CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->state_dim, ssm->state_dim, ssm->batch_size,
                                &alpha, d_H_prev, ssm->state_dim,
                                d_error_hidden_t, ssm->state_dim,
                                &alpha, ssm->d_A_grad, ssm->state_dim));
        
        // ∂L/∂H_{t-1} += (∂L/∂H_t) A
        CHECK_CUBLAS(cublasHgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                &alpha, ssm->d_A, ssm->state_dim,
                                d_error_hidden_t, ssm->state_dim,
                                &alpha, d_error_hidden_prev, ssm->state_dim));
    }
}

// CUDA kernel for AdamW update (all FP16)
__global__ void adamw_update_kernel_ssm(__half* weight, __half* grad, __half* m, __half* v,
                                        __half beta1, __half beta2, __half epsilon, __half learning_rate,
                                        __half weight_decay, __half alpha_t, int size, int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __half g = __hdiv(grad[idx], __float2half((float)total_samples));
        __half one = __float2half(1.0f);
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        __half one_minus_beta1 = __hsub(one, beta1);
        m[idx] = __hadd(__hmul(beta1, m[idx]), __hmul(one_minus_beta1, g));
        
        // v = β₂v + (1-β₂)(∂L/∂W)²
        __half one_minus_beta2 = __hsub(one, beta2);
        __half g_squared = __hmul(g, g);
        v[idx] = __hadd(__hmul(beta2, v[idx]), __hmul(one_minus_beta2, g_squared));
        
        __half update = __hdiv(__hmul(alpha_t, m[idx]), __hadd(hsqrt(v[idx]), epsilon));
        
        // W = (1-λη)W - update
        __half decay_factor = __hsub(one, __hmul(learning_rate, weight_decay));
        weight[idx] = __hsub(__hmul(weight[idx], decay_factor), update);
    }
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, __half learning_rate) {
    ssm->t++;  // Increment time step
    
    float beta1_t_f = powf(__half2float(ssm->beta1), ssm->t);
    float beta2_t_f = powf(__half2float(ssm->beta2), ssm->t);
    float alpha_t_f = __half2float(learning_rate) * sqrtf(1.0f - beta2_t_f) / (1.0f - beta1_t_f);
    __half alpha_t = __float2half(alpha_t_f);
    
    int total_samples = ssm->seq_len * ssm->batch_size;
    int block_size = 256;
    
    // Update A weights
    int A_size = ssm->state_dim * ssm->state_dim;
    int A_blocks = (A_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<A_blocks, block_size>>>(
        ssm->d_A, ssm->d_A_grad, ssm->d_A_m, ssm->d_A_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, A_size, total_samples
    );
    
    // Update B weights
    int B_size = ssm->state_dim * ssm->input_dim;
    int B_blocks = (B_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<B_blocks, block_size>>>(
        ssm->d_B, ssm->d_B_grad, ssm->d_B_m, ssm->d_B_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, B_size, total_samples
    );
    
    // Update C weights
    int C_size = ssm->output_dim * ssm->state_dim;
    int C_blocks = (C_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<C_blocks, block_size>>>(
        ssm->d_C, ssm->d_C_grad, ssm->d_C_m, ssm->d_C_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, C_size, total_samples
    );
    
    // Update D weights
    int D_size = ssm->output_dim * ssm->input_dim;
    int D_blocks = (D_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<D_blocks, block_size>>>(
        ssm->d_D, ssm->d_D_grad, ssm->d_D_m, ssm->d_D_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, D_size, total_samples
    );
}

// Function to save model weights to binary file
void save_ssm(SSM* ssm, const char* filename) {
    // Allocate temporary host memory for weights
    __half* A = (__half*)malloc(ssm->state_dim * ssm->state_dim * sizeof(__half));
    __half* B = (__half*)malloc(ssm->state_dim * ssm->input_dim * sizeof(__half));
    __half* C = (__half*)malloc(ssm->output_dim * ssm->state_dim * sizeof(__half));
    __half* D = (__half*)malloc(ssm->output_dim * ssm->input_dim * sizeof(__half));
    
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(A, ssm->d_A, ssm->state_dim * ssm->state_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B, ssm->d_B, ssm->state_dim * ssm->input_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C, ssm->d_C, ssm->output_dim * ssm->state_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D, ssm->d_D, ssm->output_dim * ssm->input_dim * sizeof(__half), cudaMemcpyDeviceToHost));

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
    fwrite(A, sizeof(__half), ssm->state_dim * ssm->state_dim, file);
    fwrite(B, sizeof(__half), ssm->state_dim * ssm->input_dim, file);
    fwrite(C, sizeof(__half), ssm->output_dim * ssm->state_dim, file);
    fwrite(D, sizeof(__half), ssm->output_dim * ssm->input_dim, file);
    
    // Save Adam state
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Also save Adam state variables
    __half* A_m = (__half*)malloc(ssm->state_dim * ssm->state_dim * sizeof(__half));
    __half* A_v = (__half*)malloc(ssm->state_dim * ssm->state_dim * sizeof(__half));
    __half* B_m = (__half*)malloc(ssm->state_dim * ssm->input_dim * sizeof(__half));
    __half* B_v = (__half*)malloc(ssm->state_dim * ssm->input_dim * sizeof(__half));
    __half* C_m = (__half*)malloc(ssm->output_dim * ssm->state_dim * sizeof(__half));
    __half* C_v = (__half*)malloc(ssm->output_dim * ssm->state_dim * sizeof(__half));
    __half* D_m = (__half*)malloc(ssm->output_dim * ssm->input_dim * sizeof(__half));
    __half* D_v = (__half*)malloc(ssm->output_dim * ssm->input_dim * sizeof(__half));
    
    CHECK_CUDA(cudaMemcpy(A_m, ssm->d_A_m, ssm->state_dim * ssm->state_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(A_v, ssm->d_A_v, ssm->state_dim * ssm->state_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_m, ssm->d_B_m, ssm->state_dim * ssm->input_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_v, ssm->d_B_v, ssm->state_dim * ssm->input_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_m, ssm->d_C_m, ssm->output_dim * ssm->state_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_v, ssm->d_C_v, ssm->output_dim * ssm->state_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_m, ssm->d_D_m, ssm->output_dim * ssm->input_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_v, ssm->d_D_v, ssm->output_dim * ssm->input_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    
    fwrite(A_m, sizeof(__half), ssm->state_dim * ssm->state_dim, file);
    fwrite(A_v, sizeof(__half), ssm->state_dim * ssm->state_dim, file);
    fwrite(B_m, sizeof(__half), ssm->state_dim * ssm->input_dim, file);
    fwrite(B_v, sizeof(__half), ssm->state_dim * ssm->input_dim, file);
    fwrite(C_m, sizeof(__half), ssm->output_dim * ssm->state_dim, file);
    fwrite(C_v, sizeof(__half), ssm->output_dim * ssm->state_dim, file);
    fwrite(D_m, sizeof(__half), ssm->output_dim * ssm->input_dim, file);
    fwrite(D_v, sizeof(__half), ssm->output_dim * ssm->input_dim, file);
    
    // Free temporary host memory
    free(A); free(B); free(C); free(D);
    free(A_m); free(A_v);
    free(B_m); free(B_v);
    free(C_m); free(C_v);
    free(D_m); free(D_v);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to load model weights from binary file
SSM* load_ssm(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle) {
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
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size, cublas_handle);
    
    // Allocate temporary host memory for weights
    __half* A = (__half*)malloc(state_dim * state_dim * sizeof(__half));
    __half* B = (__half*)malloc(state_dim * input_dim * sizeof(__half));
    __half* C = (__half*)malloc(output_dim * state_dim * sizeof(__half));
    __half* D = (__half*)malloc(output_dim * input_dim * sizeof(__half));
    
    // Load matrices
    fread(A, sizeof(__half), state_dim * state_dim, file);
    fread(B, sizeof(__half), state_dim * input_dim, file);
    fread(C, sizeof(__half), output_dim * state_dim, file);
    fread(D, sizeof(__half), output_dim * input_dim, file);
    fread(&ssm->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Load Adam state variables
    __half* A_m = (__half*)malloc(state_dim * state_dim * sizeof(__half));
    __half* A_v = (__half*)malloc(state_dim * state_dim * sizeof(__half));
    __half* B_m = (__half*)malloc(state_dim * input_dim * sizeof(__half));
    __half* B_v = (__half*)malloc(state_dim * input_dim * sizeof(__half));
    __half* C_m = (__half*)malloc(output_dim * state_dim * sizeof(__half));
    __half* C_v = (__half*)malloc(output_dim * state_dim * sizeof(__half));
    __half* D_m = (__half*)malloc(output_dim * input_dim * sizeof(__half));
    __half* D_v = (__half*)malloc(output_dim * input_dim * sizeof(__half));
    
    fread(A_m, sizeof(__half), state_dim * state_dim, file);
    fread(A_v, sizeof(__half), state_dim * state_dim, file);
    fread(B_m, sizeof(__half), state_dim * input_dim, file);
    fread(B_v, sizeof(__half), state_dim * input_dim, file);
    fread(C_m, sizeof(__half), output_dim * state_dim, file);
    fread(C_v, sizeof(__half), output_dim * state_dim, file);
    fread(D_m, sizeof(__half), output_dim * input_dim, file);
    fread(D_v, sizeof(__half), output_dim * input_dim, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A_m, A_m, state_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_v, A_v, state_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_m, B_m, state_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_v, B_v, state_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_m, C_m, output_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_v, C_v, output_dim * state_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m, D_m, output_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v, D_v, output_dim * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(A); free(B); free(C); free(D);
    free(A_m); free(A_v);
    free(B_m); free(B_v);
    free(C_m); free(C_v);
    free(D_m); free(D_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}