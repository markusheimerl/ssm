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
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->t = 0;
    ssm->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    ssm->cublas_handle = cublas_handle;
    
    // Allocate host memory for weights (local variables)
    float* A = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Initialize weights on host
    float scale_A = 0.5f / sqrtf(state_dim);
    float scale_B = 1.0f / sqrtf(input_dim);
    float scale_C = 1.0f / sqrtf(state_dim);
    float scale_D = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < state_dim * state_dim; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_A;
    }
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        B[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        C[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        D[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_D;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(float)));
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
    
    // Allocate device memory for layer outputs and working buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_layer1_preact, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_layer1_output, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_layer2_output, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error_hidden, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error_output, seq_len * batch_size * output_dim * sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    
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
    free(ssm);
}

// Reset state for new sequence
void reset_state_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_layer1_preact, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_layer1_output, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_layer2_output, 0, ssm->seq_len * ssm->batch_size * ssm->output_dim * sizeof(float)));
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel_ssm(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = input[idx];
        output[idx] = h / (1.0f + expf(-h));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel_ssm(float* grad_input, float* grad_output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-h));
        grad_input[idx] = grad_output[idx] * (sigmoid + h * sigmoid * (1.0f - sigmoid));
    }
}

// Forward pass for single timestep
void forward_pass_ssm(SSM* ssm, float* d_X_t, int timestep) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Get pointers to current timestep data (time-major format)
    float* d_H_t = &ssm->d_layer1_preact[timestep * ssm->batch_size * ssm->state_dim];
    float* d_S_t = &ssm->d_layer1_output[timestep * ssm->batch_size * ssm->state_dim];
    float* d_Y_t = &ssm->d_layer2_output[timestep * ssm->batch_size * ssm->output_dim];
    
    // H_t = X_t B^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_B, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &beta, d_H_t, ssm->state_dim));
    
    // H_t = H_t + H_{t-1} A^T
    if (timestep > 0) {
        float* d_H_prev = &ssm->d_layer1_preact[(timestep-1) * ssm->batch_size * ssm->state_dim];
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                &alpha, ssm->d_A, ssm->state_dim,
                                d_H_prev, ssm->state_dim,
                                &beta_add, d_H_t, ssm->state_dim));
    }
    
    // S_t = H_tσ(H_t)
    int block_size = 256;
    int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_S_t, d_H_t, ssm->batch_size * ssm->state_dim);
    
    // Y_t = S_t C^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->state_dim,
                            &alpha, ssm->d_C, ssm->state_dim,
                            d_S_t, ssm->state_dim,
                            &beta, d_Y_t, ssm->output_dim));
    
    // Y_t = Y_t + X_t D^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_D, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &beta_add, d_Y_t, ssm->output_dim));
}

// Calculate loss
float calculate_loss_ssm(SSM* ssm, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    int total_size = ssm->seq_len * ssm->batch_size * ssm->output_dim;
    CHECK_CUBLAS(cublasSgeam(ssm->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->output_dim, ssm->seq_len * ssm->batch_size,
                            &alpha, ssm->d_layer2_output, ssm->output_dim,
                            &beta, d_y, ssm->output_dim,
                            ssm->d_error_output, ssm->output_dim));
    CHECK_CUBLAS(cublasSdot(ssm->cublas_handle, total_size, ssm->d_error_output, 1, ssm->d_error_output, 1, &loss));
    
    return loss / total_size;
}

// Zero gradients
void zero_gradients_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_error_hidden, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
}

// Backward pass for single timestep
void backward_pass_ssm(SSM* ssm, float* d_X_t, int timestep) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Get pointers to current timestep data
    float* d_H_t = &ssm->d_layer1_preact[timestep * ssm->batch_size * ssm->state_dim];
    float* d_S_t = &ssm->d_layer1_output[timestep * ssm->batch_size * ssm->state_dim];
    float* d_error_output_t = &ssm->d_error_output[timestep * ssm->batch_size * ssm->output_dim];
    float* d_error_hidden_t = &ssm->d_error_hidden[timestep * ssm->batch_size * ssm->state_dim];
    
    // ∂L/∂C += (∂L/∂Y_t)^T S_t
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->state_dim, ssm->output_dim, ssm->batch_size,
                            &alpha, d_S_t, ssm->state_dim,
                            d_error_output_t, ssm->output_dim,
                            &beta_add, ssm->d_C_grad, ssm->state_dim));
    
    // ∂L/∂D += (∂L/∂Y_t)^T X_t
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->input_dim, ssm->output_dim, ssm->batch_size,
                            &alpha, d_X_t, ssm->input_dim,
                            d_error_output_t, ssm->output_dim,
                            &beta_add, ssm->d_D_grad, ssm->input_dim));
    
    // ∂L/∂S_t = (∂L/∂Y_t) C
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
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
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->input_dim, ssm->state_dim, ssm->batch_size,
                            &alpha, d_X_t, ssm->input_dim,
                            d_error_hidden_t, ssm->state_dim,
                            &beta_add, ssm->d_B_grad, ssm->input_dim));
    
    // Propagate error to previous timestep
    if (timestep > 0) {
        float* d_H_prev = &ssm->d_layer1_preact[(timestep-1) * ssm->batch_size * ssm->state_dim];
        float* d_error_hidden_prev = &ssm->d_error_hidden[(timestep-1) * ssm->batch_size * ssm->state_dim];
        
        // ∂L/∂A += (∂L/∂H_t)^T H_{t-1}
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->state_dim, ssm->state_dim, ssm->batch_size,
                                &alpha, d_H_prev, ssm->state_dim,
                                d_error_hidden_t, ssm->state_dim,
                                &beta_add, ssm->d_A_grad, ssm->state_dim));
        
        // ∂L/∂H_{t-1} += (∂L/∂H_t) A
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                &alpha, ssm->d_A, ssm->state_dim,
                                d_error_hidden_t, ssm->state_dim,
                                &beta_add, d_error_hidden_prev, ssm->state_dim));
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_ssm(float* weight, float* grad, float* m, float* v,
                                        float beta1, float beta2, float epsilon, float learning_rate,
                                        float weight_decay, float alpha_t, int size, int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / total_samples;
        
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
    ssm->t++;  // Increment time step
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
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
    float* A = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    float* B = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* C = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    // Copy weights from device to host
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
    
    // Save Adam state
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Also save Adam state variables
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
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state variables
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
    free(A_m); free(A_v);
    free(B_m); free(B_v);
    free(C_m); free(C_v);
    free(D_m); free(D_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}