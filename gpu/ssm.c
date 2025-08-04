#include "ssm.h"

// CUDA kernel for creating skew-symmetric matrix from upper triangular parameters
__global__ void create_skew_symmetric_kernel(float* A_skew_full, const float* A_skew_params, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;
    
    if (idx < total_elements) {
        int i = idx / n;
        int j = idx % n;
        
        if (i == j) {
            A_skew_full[idx] = 0.0f;  // Diagonal elements are zero
        } else if (i < j) {
            // Upper triangle: find parameter index
            int param_idx = i * n - (i * (i + 1)) / 2 + (j - i - 1);
            A_skew_full[idx] = A_skew_params[param_idx];
        } else {
            // Lower triangle: negative of corresponding upper triangle
            int param_idx = j * n - (j * (j + 1)) / 2 + (i - j - 1);
            A_skew_full[idx] = -A_skew_params[param_idx];
        }
    }
}

// CUDA kernel for simplified matrix exponential (Padé approximation)
__global__ void matrix_exponential_pade_kernel(float* exp_A, const float* A_skew_full, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;
    
    if (idx < total_elements) {
        int i = idx / n;
        int j = idx % n;
        
        // For small matrices, use first-order approximation: exp(A) ≈ I + A
        if (i == j) {
            exp_A[idx] = 1.0f + A_skew_full[idx];  // I + A
        } else {
            exp_A[idx] = A_skew_full[idx];
        }
    }
}

// CUDA kernel for matrix exponential gradient computation
__global__ void matrix_exponential_gradient_kernel(float* grad_A_skew, const float* grad_exp_A, int n, int skew_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < skew_params) {
        // Convert linear index to upper triangular (i,j) coordinates
        int param_idx = 0;
        int i = 0, j = 1;
        
        while (param_idx < idx) {
            j++;
            if (j >= n) {
                i++;
                j = i + 1;
            }
            param_idx++;
        }
        
        // Gradient w.r.t. A_skew[param] = grad_exp_A[i,j] - grad_exp_A[j,i]
        grad_A_skew[idx] = grad_exp_A[i * n + j] - grad_exp_A[j * n + i];
    }
}

// GPU utility functions
void create_skew_symmetric_gpu(float* d_A_skew_full, const float* d_A_skew_params, int n) {
    int total_elements = n * n;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    create_skew_symmetric_kernel<<<grid_size, block_size>>>(d_A_skew_full, d_A_skew_params, n);
    CHECK_CUDA(cudaGetLastError());
}

void matrix_exponential_pade_gpu(float* d_exp_A, const float* d_A_skew_full, int n) {
    int total_elements = n * n;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    matrix_exponential_pade_kernel<<<grid_size, block_size>>>(d_exp_A, d_A_skew_full, n);
    CHECK_CUDA(cudaGetLastError());
}

void matrix_exponential_gradient_gpu(float* d_grad_A_skew, const float* d_grad_exp_A, int n) {
    int skew_params = n * (n - 1) / 2;
    int block_size = 256;
    int grid_size = (skew_params + block_size - 1) / block_size;
    
    matrix_exponential_gradient_kernel<<<grid_size, block_size>>>(d_grad_A_skew, d_grad_exp_A, n, skew_params);
    CHECK_CUDA(cudaGetLastError());
}

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
    
    // Calculate number of skew-symmetric parameters
    int skew_params = state_dim * (state_dim - 1) / 2;
    
    // Allocate host memory for initialization
    float* A_skew = (float*)malloc(skew_params * sizeof(float));
    float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Initialize B, C, D matrices
    float scale_B = 0.5f / sqrtf(input_dim);
    float scale_C = 0.5f / sqrtf(state_dim);
    float scale_D = 0.1f / sqrtf(input_dim);
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        B[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        C[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        D[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_D;
    }
    
    // Initialize A_skew parameters with small random values
    float scale_A_skew = 0.1f / sqrtf(state_dim);
    for (int i = 0; i < skew_params; i++) {
        A_skew[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_A_skew;
    }
    
    // Allocate device memory for matrices
    CHECK_CUDA(cudaMalloc(&ssm->d_A_skew, skew_params * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_orthogonal, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A_skew_grad, skew_params * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_orthogonal_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ssm->d_A_skew_m, skew_params * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_skew_v, skew_params * sizeof(float)));
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
    CHECK_CUDA(cudaMemcpy(ssm->d_A_skew, A_skew, skew_params * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ssm->d_A_skew_m, 0, skew_params * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_skew_v, 0, skew_params * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    
    // Compute initial A_orthogonal from A_skew on GPU
    float* d_A_skew_full;
    CHECK_CUDA(cudaMalloc(&d_A_skew_full, state_dim * state_dim * sizeof(float)));
    create_skew_symmetric_gpu(d_A_skew_full, ssm->d_A_skew, state_dim);
    matrix_exponential_pade_gpu(ssm->d_A_orthogonal, d_A_skew_full, state_dim);
    CHECK_CUDA(cudaFree(d_A_skew_full));
    
    // Free host memory
    free(A_skew);
    free(B);
    free(C);
    free(D);
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    // Free device memory
    cudaFree(ssm->d_A_skew); cudaFree(ssm->d_A_orthogonal); cudaFree(ssm->d_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);
    cudaFree(ssm->d_A_skew_grad); cudaFree(ssm->d_A_orthogonal_grad); cudaFree(ssm->d_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_A_skew_m); cudaFree(ssm->d_A_skew_v); cudaFree(ssm->d_B_m); cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m); cudaFree(ssm->d_C_v); cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);
    cudaFree(ssm->d_states); cudaFree(ssm->d_predictions); cudaFree(ssm->d_error); 
    cudaFree(ssm->d_state_error); cudaFree(ssm->d_state_outputs);
    
    // Destroy cuBLAS handle
    cublasDestroy(ssm->cublas_handle);
    
    free(ssm);
}

// Reset SSM state to zero
void reset_state_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
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
void forward_pass_ssm(SSM* ssm, float* d_X_t, int timestep) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Get pointers to current timestep state
    float* d_h_prev = (timestep > 0) ? ssm->d_states + (timestep - 1) * ssm->batch_size * ssm->state_dim : NULL;
    float* d_h_t = ssm->d_states + timestep * ssm->batch_size * ssm->state_dim;
    float* d_o_t = ssm->d_state_outputs + timestep * ssm->batch_size * ssm->state_dim;
    float* d_y_t = ssm->d_predictions + timestep * ssm->batch_size * ssm->output_dim;
    
    // H_t = X_t B^T + H_{t-1} A^T
    // H_t = X_t B^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_B, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &beta, d_h_t, ssm->state_dim));
    
    // H_t += H_{t-1} A^T
    if (timestep > 0) {
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                &alpha, ssm->d_A_orthogonal, ssm->state_dim,
                                d_h_prev, ssm->state_dim,
                                &beta_add, d_h_t, ssm->state_dim));
    }
    
    // O_t = H_t σ(H_t)
    int block_size = 256;
    int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_o_t, d_h_t, ssm->batch_size * ssm->state_dim);
    
    // Y_t = O_t C^T + X_t D^T
    // Y_t = O_t C^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->state_dim,
                            &alpha, ssm->d_C, ssm->state_dim,
                            d_o_t, ssm->state_dim,
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
    int skew_params = ssm->state_dim * (ssm->state_dim - 1) / 2;
    CHECK_CUDA(cudaMemset(ssm->d_A_skew_grad, 0, skew_params * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_orthogonal_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float)));
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
        
        // ∂L/∂C += (∂L/∂Y_t)^T O_t
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->state_dim, ssm->output_dim, ssm->batch_size,
                                &alpha, d_o_t, ssm->state_dim,
                                d_dy_t, ssm->output_dim,
                                &beta_add, ssm->d_C_grad, ssm->state_dim));
        
        // ∂L/∂D += (∂L/∂Y_t)^T X_t
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->input_dim, ssm->output_dim, ssm->batch_size,
                                &alpha, d_X_t, ssm->input_dim,
                                d_dy_t, ssm->output_dim,
                                &beta_add, ssm->d_D_grad, ssm->input_dim));
        
        // ∂L/∂O_t = (∂L/∂Y_t)C
        float* d_do_t = d_o_t; // reuse buffer
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->output_dim,
                                &alpha, ssm->d_C, ssm->state_dim,
                                d_dy_t, ssm->output_dim,
                                &beta, d_do_t, ssm->state_dim));
        
        // ∂L/∂H_t = ∂L/∂O_t ⊙ [σ(H_t) + H_t σ(H_t)(1-σ(H_t))]
        int block_size = 256;
        int num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
        swish_backward_kernel_ssm<<<num_blocks, block_size>>>(d_dh_t, d_do_t, d_h_t, ssm->batch_size * ssm->state_dim);
        
        // ∂L/∂H_t += (∂L/∂H_{t+1})A
        if (t < ssm->seq_len - 1) {
            float* d_dh_next = ssm->d_state_error + (t+1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                    &alpha, ssm->d_A_orthogonal, ssm->state_dim,
                                    d_dh_next, ssm->state_dim,
                                    &beta_add, d_dh_t, ssm->state_dim));
        }
        
        // ∂L/∂B += (∂L/∂H_t)^T X_t
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->input_dim, ssm->state_dim, ssm->batch_size,
                                &alpha, d_X_t, ssm->input_dim,
                                d_dh_t, ssm->state_dim,
                                &beta_add, ssm->d_B_grad, ssm->input_dim));
        
        // ∂L/∂A += (∂L/∂H_t)^T H_{t-1}
        if (t > 0) {
            float* d_h_prev = ssm->d_states + (t-1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                    ssm->state_dim, ssm->state_dim, ssm->batch_size,
                                    &alpha, d_h_prev, ssm->state_dim,
                                    d_dh_t, ssm->state_dim,
                                    &beta_add, ssm->d_A_orthogonal_grad, ssm->state_dim));
        }
    }
    
    // Transform gradient from A_orthogonal space to A_skew space
    // For now, use simplified transformation (should be proper chain rule through matrix exponential)
    matrix_exponential_gradient_gpu(ssm->d_A_skew_grad, ssm->d_A_orthogonal_grad, ssm->state_dim);
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
    
    // Update A_skew
    int skew_params = ssm->state_dim * (ssm->state_dim - 1) / 2;
    int A_blocks = (skew_params + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<A_blocks, block_size>>>(
        ssm->d_A_skew, ssm->d_A_skew_grad, ssm->d_A_skew_m, ssm->d_A_skew_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, skew_params, ssm->batch_size
    );
    
    // Recompute A_orthogonal from updated A_skew
    compute_orthogonal_matrix_gpu(ssm);
    
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
    int skew_params = ssm->state_dim * (ssm->state_dim - 1) / 2;
    float* A_skew = (float*)malloc(skew_params * sizeof(float));
    float* B = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* C = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(A_skew, ssm->d_A_skew, skew_params * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B, ssm->d_B, ssm->state_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C, ssm->d_C, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D, ssm->d_D, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(A_skew); free(B); free(C); free(D);
        return;
    }
    
    // Save dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->seq_len, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
    // Save matrices
    fwrite(A_skew, sizeof(float), skew_params, file);
    fwrite(B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Save Adam state
    float* A_skew_m = (float*)malloc(skew_params * sizeof(float));
    float* A_skew_v = (float*)malloc(skew_params * sizeof(float));
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
