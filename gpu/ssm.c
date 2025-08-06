#include "ssm.h"

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
    CHECK_CUBLAS(cublasSetMathMode(ssm->cublas_handle, CUBLAS_TENSOR_OP_MATH));
    
    // Allocate host memory for initialization
    float* A = (float*)calloc(state_dim * state_dim, sizeof(float));
    float* W_B = (float*)malloc(state_dim * input_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Initialize A, W_B, C, D matrices
    float scale_A = 0.1f / sqrtf(state_dim);
    float scale_W_B = 0.1f / sqrtf(input_dim * input_dim);
    float scale_C = 0.5f / sqrtf(state_dim);
    float scale_D = 0.1f / sqrtf(input_dim);

    for (int i = 0; i < state_dim * state_dim; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_A;
    }
    
    for (int i = 0; i < state_dim * input_dim * input_dim; i++) {
        W_B[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        C[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        D[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_D;
    }
    
    // Allocate device memory for matrices
    CHECK_CUDA(cudaMalloc(&ssm->d_A, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B, state_dim * input_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B_grad, state_dim * input_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ssm->d_A_m, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_v, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B_m, state_dim * input_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B_v, state_dim * input_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_m, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_v, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_m, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_v, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for helper arrays
    CHECK_CUDA(cudaMalloc(&ssm->d_states, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_X_outer, seq_len * batch_size * input_dim * input_dim * sizeof(float)));
    
    // Copy initialized matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B, W_B, state_dim * input_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_B_m, 0, state_dim * input_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_B_v, 0, state_dim * input_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    
    // Free host memory
    free(A);
    free(W_B);
    free(C);
    free(D);
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    // Free device memory
    cudaFree(ssm->d_A); cudaFree(ssm->d_W_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);
    cudaFree(ssm->d_A_grad); cudaFree(ssm->d_W_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_A_m); cudaFree(ssm->d_A_v); cudaFree(ssm->d_W_B_m); cudaFree(ssm->d_W_B_v);
    cudaFree(ssm->d_C_m); cudaFree(ssm->d_C_v); cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);
    cudaFree(ssm->d_states); cudaFree(ssm->d_predictions); cudaFree(ssm->d_error); 
    cudaFree(ssm->d_state_error); cudaFree(ssm->d_X_outer);
    
    // Destroy cuBLAS handle
    cublasDestroy(ssm->cublas_handle);
    
    free(ssm);
}

// Reset SSM state to zero
void reset_state_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
}

// CUDA kernel for computing outer products
__global__ void compute_outer_products_kernel(float* X_outer, const float* X, 
                                             int batch_size, int input_dim) {
    int b = blockIdx.x;  // batch index
    int i = threadIdx.x; // first input dim
    int j = threadIdx.y; // second input dim
    
    if (b < batch_size && i < input_dim && j < input_dim) {
        int x_idx = b * input_dim;
        int out_idx = b * input_dim * input_dim + i * input_dim + j;
        X_outer[out_idx] = X[x_idx + i] * X[x_idx + j];
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
void forward_pass_ssm(SSM* ssm, float* d_X_t, int timestep) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Get pointers to current timestep state
    float* d_h_prev = (timestep > 0) ? ssm->d_states + (timestep - 1) * ssm->batch_size * ssm->state_dim : NULL;
    float* d_h_t = ssm->d_states + timestep * ssm->batch_size * ssm->state_dim;
    float* d_y_t = ssm->d_predictions + timestep * ssm->batch_size * ssm->output_dim;
    
    // Compute input-dependent B using bilinear form: H_t[i] = X_t^T W_B[i] X_t
    // Step 1: Compute outer products X_t ⊗ X_t for each batch element
    float* d_X_outer_t = ssm->d_X_outer + timestep * ssm->batch_size * ssm->input_dim * ssm->input_dim;
    
    // Launch kernel to compute all outer products at once
    dim3 block(ssm->input_dim, ssm->input_dim);
    dim3 grid(ssm->batch_size);
    compute_outer_products_kernel<<<grid, block>>>(d_X_outer_t, d_X_t, ssm->batch_size, ssm->input_dim);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Step 2: H_t = X_outer * W_B^T
    // d_X_outer_t: batch_size x (input_dim * input_dim)
    // d_W_B: state_dim x (input_dim * input_dim)
    // Result: batch_size x state_dim
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim * ssm->input_dim,
                            &alpha, ssm->d_W_B, ssm->input_dim * ssm->input_dim,
                            d_X_outer_t, ssm->input_dim * ssm->input_dim,
                            &beta, d_h_t, ssm->state_dim));
    
    // H_t += H_{t-1} A^T
    if (timestep > 0) {
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                &alpha, ssm->d_A, ssm->state_dim,
                                d_h_prev, ssm->state_dim,
                                &beta_add, d_h_t, ssm->state_dim));
    }
    
    // Y_t = H_t C^T + X_t D^T (no activation on H_t)
    // Y_t = H_t C^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->state_dim,
                            &alpha, ssm->d_C, ssm->state_dim,
                            d_h_t, ssm->state_dim,
                            &beta, d_y_t, ssm->output_dim));
    
    // Y_t += X_t D^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_D, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &beta_add, d_y_t, ssm->output_dim));
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
    CHECK_CUDA(cudaMemset(ssm->d_W_B_grad, 0, ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float)));
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
        float* d_dy_t = ssm->d_error + t * ssm->batch_size * ssm->output_dim;
        float* d_dh_t = ssm->d_state_error + t * ssm->batch_size * ssm->state_dim;
        float* d_X_outer_t = ssm->d_X_outer + t * ssm->batch_size * ssm->input_dim * ssm->input_dim;
        
        // ∂L/∂C += (∂L/∂Y_t)^T H_t (direct since no activation)
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->state_dim, ssm->output_dim, ssm->batch_size,
                                &alpha, d_h_t, ssm->state_dim,
                                d_dy_t, ssm->output_dim,
                                &beta_add, ssm->d_C_grad, ssm->state_dim));
        
        // ∂L/∂D += (∂L/∂Y_t)^T X_t
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->input_dim, ssm->output_dim, ssm->batch_size,
                                &alpha, d_X_t, ssm->input_dim,
                                d_dy_t, ssm->output_dim,
                                &beta_add, ssm->d_D_grad, ssm->input_dim));
        
        // ∂L/∂H_t = (∂L/∂Y_t)C (direct since no activation)
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->output_dim,
                                &alpha, ssm->d_C, ssm->state_dim,
                                d_dy_t, ssm->output_dim,
                                &beta, d_dh_t, ssm->state_dim));
        
        // ∂L/∂H_t += (∂L/∂H_{t+1})A
        if (t < ssm->seq_len - 1) {
            float* d_dh_next = ssm->d_state_error + (t+1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                    &alpha, ssm->d_A, ssm->state_dim,
                                    d_dh_next, ssm->state_dim,
                                    &beta_add, d_dh_t, ssm->state_dim));
        }
        
        // ∂L/∂W_B += ∂L/∂H_t^T * X_outer_t
        // d_dh_t: batch_size x state_dim -> transpose to state_dim x batch_size
        // d_X_outer_t: batch_size x (input_dim * input_dim)
        // d_W_B_grad: state_dim x (input_dim * input_dim)
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->input_dim * ssm->input_dim, ssm->state_dim, ssm->batch_size,
                                &alpha, d_X_outer_t, ssm->input_dim * ssm->input_dim,
                                d_dh_t, ssm->state_dim,
                                &beta_add, ssm->d_W_B_grad, ssm->input_dim * ssm->input_dim));
        
        // ∂L/∂A += (∂L/∂H_t)^T H_{t-1}
        if (t > 0) {
            float* d_h_prev = ssm->d_states + (t-1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                    ssm->state_dim, ssm->state_dim, ssm->batch_size,
                                    &alpha, d_h_prev, ssm->state_dim,
                                    d_dh_t, ssm->state_dim,
                                    &beta_add, ssm->d_A_grad, ssm->state_dim));
        }
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_ssm(float* weight, float* grad, float* m, float* v,
                                        float beta1, float beta2, float epsilon, float learning_rate,
                                        float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)g
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)g²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η·m̂/√v̂
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
    
    // Update W_B
    int W_B_size = ssm->state_dim * ssm->input_dim * ssm->input_dim;
    int W_B_blocks = (W_B_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<W_B_blocks, block_size>>>(
        ssm->d_W_B, ssm->d_W_B_grad, ssm->d_W_B_m, ssm->d_W_B_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, W_B_size, ssm->batch_size
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
    float* W_B = (float*)malloc(ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float));
    float* C = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(A, ssm->d_A, ssm->state_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W_B, ssm->d_W_B, ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C, ssm->d_C, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D, ssm->d_D, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(A); free(W_B); free(C); free(D);
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
    fwrite(W_B, sizeof(float), ssm->state_dim * ssm->input_dim * ssm->input_dim, file);
    fwrite(C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Save Adam state
    float* A_m = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    float* A_v = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    float* W_B_m = (float*)malloc(ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float));
    float* W_B_v = (float*)malloc(ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float));
    float* C_m = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* C_v = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D_m = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    float* D_v = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(A_m, ssm->d_A_m, ssm->state_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(A_v, ssm->d_A_v, ssm->state_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W_B_m, ssm->d_W_B_m, ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W_B_v, ssm->d_W_B_v, ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_m, ssm->d_C_m, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_v, ssm->d_C_v, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_m, ssm->d_D_m, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_v, ssm->d_D_v, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(A_m, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(A_v, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(W_B_m, sizeof(float), ssm->state_dim * ssm->input_dim * ssm->input_dim, file);
    fwrite(W_B_v, sizeof(float), ssm->state_dim * ssm->input_dim * ssm->input_dim, file);
    fwrite(C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Free temporary host memory
    free(A); free(W_B); free(C); free(D);
    free(A_m); free(A_v); free(W_B_m); free(W_B_v);
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
    float* W_B = (float*)malloc(state_dim * input_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Load matrices
    fread(A, sizeof(float), state_dim * state_dim, file);
    fread(W_B, sizeof(float), state_dim * input_dim * input_dim, file);
    fread(C, sizeof(float), output_dim * state_dim, file);
    fread(D, sizeof(float), output_dim * input_dim, file);
    
    fread(&ssm->t, sizeof(int), 1, file);
    
    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B, W_B, state_dim * input_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    float* A_m = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* A_v = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* W_B_m = (float*)malloc(state_dim * input_dim * input_dim * sizeof(float));
    float* W_B_v = (float*)malloc(state_dim * input_dim * input_dim * sizeof(float));
    float* C_m = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* C_v = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D_m = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* D_v = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    fread(A_m, sizeof(float), state_dim * state_dim, file);
    fread(A_v, sizeof(float), state_dim * state_dim, file);
    fread(W_B_m, sizeof(float), state_dim * input_dim * input_dim, file);
    fread(W_B_v, sizeof(float), state_dim * input_dim * input_dim, file);
    fread(C_m, sizeof(float), output_dim * state_dim, file);
    fread(C_v, sizeof(float), output_dim * state_dim, file);
    fread(D_m, sizeof(float), output_dim * input_dim, file);
    fread(D_v, sizeof(float), output_dim * input_dim, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A_m, A_m, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_v, A_v, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B_m, W_B_m, state_dim * input_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B_v, W_B_v, state_dim * input_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_m, C_m, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_v, C_v, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m, D_m, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v, D_v, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(A); free(W_B); free(C); free(D);
    free(A_m); free(A_v); free(W_B_m); free(W_B_v);
    free(C_m); free(C_v); free(D_m); free(D_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}