#include "ssm.h"

// Initialize the doubly-bilinear state space model
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
    
    // Calculate matrix sizes
    int A_size = state_dim * state_dim;
    int W_B_size = state_dim * input_dim * input_dim;
    int W_C_size = output_dim * state_dim * state_dim;
    int D_size = output_dim * input_dim;
    
    // Allocate device memory for matrices
    CHECK_CUDA(cudaMalloc(&ssm->d_A, A_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B, W_B_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_C, W_C_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, D_size * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, A_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B_grad, W_B_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_C_grad, W_C_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, D_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ssm->d_A_m, A_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_v, A_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B_m, W_B_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_B_v, W_B_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_C_m, W_C_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_C_v, W_C_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_m, D_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_v, D_size * sizeof(float)));
    
    // Allocate device memory for working buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_states, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_X_outer, seq_len * batch_size * input_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_H_outer, seq_len * batch_size * state_dim * state_dim * sizeof(float)));
    
    // Initialize matrices on host
    float* h_A = (float*)calloc(A_size, sizeof(float));
    float* h_W_B = (float*)malloc(W_B_size * sizeof(float));
    float* h_W_C = (float*)malloc(W_C_size * sizeof(float));
    float* h_D = (float*)malloc(D_size * sizeof(float));
    
    // Xavier initialization with appropriate scaling
    float scale_A = 0.1f / sqrtf(state_dim);
    float scale_W_B = 0.1f / sqrtf(input_dim * input_dim);
    float scale_W_C = 0.1f / sqrtf(state_dim * state_dim);
    float scale_D = 0.1f / sqrtf(input_dim);
    
    for (int i = 0; i < A_size; i++) {
        h_A[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_A;
    }
    for (int i = 0; i < W_B_size; i++) {
        h_W_B[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W_B;
    }
    for (int i = 0; i < W_C_size; i++) {
        h_W_C[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W_C;
    }
    for (int i = 0; i < D_size; i++) {
        h_D[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_D;
    }
    
    // Copy initialized matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, h_A, A_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B, h_W_B, W_B_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_C, h_W_C, W_C_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, h_D, D_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, A_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, A_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_B_m, 0, W_B_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_B_v, 0, W_B_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_C_m, 0, W_C_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_C_v, 0, W_C_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, D_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, D_size * sizeof(float)));
    
    // Free host memory
    free(h_A); free(h_W_B); free(h_W_C); free(h_D);
    
    return ssm;
}

void free_ssm(SSM* ssm) {
    if (!ssm) return;
    
    // Free matrices
    cudaFree(ssm->d_A); cudaFree(ssm->d_W_B); cudaFree(ssm->d_W_C); cudaFree(ssm->d_D);
    
    // Free gradients
    cudaFree(ssm->d_A_grad); cudaFree(ssm->d_W_B_grad); 
    cudaFree(ssm->d_W_C_grad); cudaFree(ssm->d_D_grad);
    
    // Free Adam state
    cudaFree(ssm->d_A_m); cudaFree(ssm->d_A_v);
    cudaFree(ssm->d_W_B_m); cudaFree(ssm->d_W_B_v);
    cudaFree(ssm->d_W_C_m); cudaFree(ssm->d_W_C_v);
    cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);
    
    // Free working buffers
    cudaFree(ssm->d_states); cudaFree(ssm->d_predictions); cudaFree(ssm->d_error);
    cudaFree(ssm->d_state_error); cudaFree(ssm->d_X_outer); cudaFree(ssm->d_H_outer);
    
    cublasDestroy(ssm->cublas_handle);
    free(ssm);
}

void reset_state_ssm(SSM* ssm) {
    CHECK_CUDA(cudaMemset(ssm->d_states, 0, 
                         ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
}

void forward_pass_ssm(SSM* ssm, float* d_X_t, int timestep) {
    const float alpha = 1.0f, beta = 0.0f, beta_add = 1.0f;
    
    // Get pointers for current timestep
    float* d_h_prev = (timestep > 0) ? ssm->d_states + (timestep - 1) * ssm->batch_size * ssm->state_dim : NULL;
    float* d_h_t = ssm->d_states + timestep * ssm->batch_size * ssm->state_dim;
    float* d_y_t = ssm->d_predictions + timestep * ssm->batch_size * ssm->output_dim;
    float* d_X_outer_t = ssm->d_X_outer + timestep * ssm->batch_size * ssm->input_dim * ssm->input_dim;
    float* d_H_outer_t = ssm->d_H_outer + timestep * ssm->batch_size * ssm->state_dim * ssm->state_dim;
    
    // Step 1: Compute X_t ⊗ X_t (input outer products)
    CHECK_CUBLAS(cublasSgemmStridedBatched(ssm->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_T,
                                          ssm->input_dim, ssm->input_dim, 1,
                                          &alpha,
                                          d_X_t, ssm->input_dim, ssm->input_dim,
                                          d_X_t, ssm->input_dim, ssm->input_dim,
                                          &beta,
                                          d_X_outer_t, ssm->input_dim, ssm->input_dim * ssm->input_dim,
                                          ssm->batch_size));
    
    // Step 2: H_t = (X_t ⊗ X_t) W_B^T + H_{t-1} A^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim * ssm->input_dim,
                            &alpha, ssm->d_W_B, ssm->input_dim * ssm->input_dim,
                            d_X_outer_t, ssm->input_dim * ssm->input_dim,
                            &beta, d_h_t, ssm->state_dim));
    
    // Add temporal connection: H_t += H_{t-1} A^T
    if (timestep > 0) {
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                &alpha, ssm->d_A, ssm->state_dim,
                                d_h_prev, ssm->state_dim,
                                &beta_add, d_h_t, ssm->state_dim));
    }
    
    // Step 3: Compute H_t ⊗ H_t (state outer products)
    CHECK_CUBLAS(cublasSgemmStridedBatched(ssm->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_T,
                                          ssm->state_dim, ssm->state_dim, 1,
                                          &alpha,
                                          d_h_t, ssm->state_dim, ssm->state_dim,
                                          d_h_t, ssm->state_dim, ssm->state_dim,
                                          &beta,
                                          d_H_outer_t, ssm->state_dim, ssm->state_dim * ssm->state_dim,
                                          ssm->batch_size));
    
    // Step 4: Y_t = (H_t ⊗ H_t) W_C^T + X_t D^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->state_dim * ssm->state_dim,
                            &alpha, ssm->d_W_C, ssm->state_dim * ssm->state_dim,
                            d_H_outer_t, ssm->state_dim * ssm->state_dim,
                            &beta, d_y_t, ssm->output_dim));
    
    // Add skip connection: Y_t += X_t D^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_D, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &beta_add, d_y_t, ssm->output_dim));
}

__global__ void calc_error_kernel_ssm(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

float calculate_loss_ssm(SSM* ssm, float* d_y) {
    int total_size = ssm->seq_len * ssm->batch_size * ssm->output_dim;
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;

    calc_error_kernel_ssm<<<num_blocks, block_size>>>(
        ssm->d_error, ssm->d_predictions, d_y, total_size);

    float loss;
    CHECK_CUBLAS(cublasSdot(ssm->cublas_handle, total_size, ssm->d_error, 1, ssm->d_error, 1, &loss));
    return loss / total_size;
}

void zero_gradients_ssm(SSM* ssm) {
    int A_size = ssm->state_dim * ssm->state_dim;
    int W_B_size = ssm->state_dim * ssm->input_dim * ssm->input_dim;
    int W_C_size = ssm->output_dim * ssm->state_dim * ssm->state_dim;
    int D_size = ssm->output_dim * ssm->input_dim;
    
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, A_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_B_grad, 0, W_B_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_C_grad, 0, W_C_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, D_size * sizeof(float)));
}

void backward_pass_ssm(SSM* ssm, float* d_X) {
    const float alpha = 1.0f, beta = 0.0f, beta_add = 1.0f;
    
    // Clear state errors
    CHECK_CUDA(cudaMemset(ssm->d_state_error, 0, 
                         ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
    
    for (int t = ssm->seq_len - 1; t >= 0; t--) {
        float* d_X_t = d_X + t * ssm->batch_size * ssm->input_dim;
        float* d_h_t = ssm->d_states + t * ssm->batch_size * ssm->state_dim;
        float* d_dy_t = ssm->d_error + t * ssm->batch_size * ssm->output_dim;
        float* d_dh_t = ssm->d_state_error + t * ssm->batch_size * ssm->state_dim;
        float* d_X_outer_t = ssm->d_X_outer + t * ssm->batch_size * ssm->input_dim * ssm->input_dim;
        float* d_H_outer_t = ssm->d_H_outer + t * ssm->batch_size * ssm->state_dim * ssm->state_dim;
        
        // ∂L/∂W_C += (∂L/∂Y_t)^T (H_t ⊗ H_t)
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->state_dim * ssm->state_dim, ssm->output_dim, ssm->batch_size,
                                &alpha, d_H_outer_t, ssm->state_dim * ssm->state_dim,
                                d_dy_t, ssm->output_dim,
                                &beta_add, ssm->d_W_C_grad, ssm->state_dim * ssm->state_dim));
        
        // ∂L/∂D += (∂L/∂Y_t)^T X_t
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                ssm->input_dim, ssm->output_dim, ssm->batch_size,
                                &alpha, d_X_t, ssm->input_dim,
                                d_dy_t, ssm->output_dim,
                                &beta_add, ssm->d_D_grad, ssm->input_dim));
        
        // ∂L/∂(H_t ⊗ H_t) = (∂L/∂Y_t) W_C
        float* d_H_outer_grad = d_H_outer_t; // Reuse buffer for gradient
        CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                ssm->state_dim * ssm->state_dim, ssm->batch_size, ssm->output_dim,
                                &alpha, ssm->d_W_C, ssm->state_dim * ssm->state_dim,
                                d_dy_t, ssm->output_dim,
                                &beta, d_H_outer_grad, ssm->state_dim * ssm->state_dim));
        
        // ∂L/∂H_t from outer product using GEMV calls
        for (int b = 0; b < ssm->batch_size; b++) {
            float* h_b = d_h_t + b * ssm->state_dim;
            float* grad_outer_b = d_H_outer_grad + b * ssm->state_dim * ssm->state_dim;
            float* dh_b = d_dh_t + b * ssm->state_dim;
            
            // First term: grad_outer @ H (treating grad_outer as state_dim x state_dim matrix)
            CHECK_CUBLAS(cublasSgemv(ssm->cublas_handle, CUBLAS_OP_N,
                                    ssm->state_dim, ssm->state_dim,
                                    &alpha, grad_outer_b, ssm->state_dim,
                                    h_b, 1,
                                    &beta, dh_b, 1));
            
            // Second term: grad_outer^T @ H (transpose the matrix)
            CHECK_CUBLAS(cublasSgemv(ssm->cublas_handle, CUBLAS_OP_T,
                                    ssm->state_dim, ssm->state_dim,
                                    &alpha, grad_outer_b, ssm->state_dim,
                                    h_b, 1,
                                    &beta_add, dh_b, 1));
        }
        
        // ∂L/∂H_t += (∂L/∂H_{t+1}) A (temporal gradient)
        if (t < ssm->seq_len - 1) {
            float* d_dh_next = ssm->d_state_error + (t+1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    ssm->state_dim, ssm->batch_size, ssm->state_dim,
                                    &alpha, ssm->d_A, ssm->state_dim,
                                    d_dh_next, ssm->state_dim,
                                    &beta_add, d_dh_t, ssm->state_dim));
        }
        
        // ∂L/∂W_B += (∂L/∂H_t)^T (X_t ⊗ X_t)
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

__global__ void adamw_update_kernel_ssm(float* weight, float* grad, float* m, float* v,
                                        float beta1, float beta2, float epsilon, float learning_rate,
                                        float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update each matrix
    int sizes[] = {
        ssm->state_dim * ssm->state_dim,                    // A
        ssm->state_dim * ssm->input_dim * ssm->input_dim,   // W_B
        ssm->output_dim * ssm->state_dim * ssm->state_dim,  // W_C
        ssm->output_dim * ssm->input_dim                    // D
    };
    
    float* weights[] = {ssm->d_A, ssm->d_W_B, ssm->d_W_C, ssm->d_D};
    float* grads[] = {ssm->d_A_grad, ssm->d_W_B_grad, ssm->d_W_C_grad, ssm->d_D_grad};
    float* ms[] = {ssm->d_A_m, ssm->d_W_B_m, ssm->d_W_C_m, ssm->d_D_m};
    float* vs[] = {ssm->d_A_v, ssm->d_W_B_v, ssm->d_W_C_v, ssm->d_D_v};
    
    for (int i = 0; i < 4; i++) {
        int num_blocks = (sizes[i] + block_size - 1) / block_size;
        adamw_update_kernel_ssm<<<num_blocks, block_size>>>(
            weights[i], grads[i], ms[i], vs[i],
            ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
            alpha_t, sizes[i], ssm->batch_size);
    }
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
    fwrite(&ssm->seq_len, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Calculate sizes
    int A_size = ssm->state_dim * ssm->state_dim;
    int W_B_size = ssm->state_dim * ssm->input_dim * ssm->input_dim;
    int W_C_size = ssm->output_dim * ssm->state_dim * ssm->state_dim;
    int D_size = ssm->output_dim * ssm->input_dim;
    
    // Allocate temporary host memory for largest matrix
    float* temp = (float*)malloc(W_C_size * sizeof(float));
    
    // Save matrices
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_A, A_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), A_size, file);
    
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_W_B, W_B_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), W_B_size, file);
    
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_W_C, W_C_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), W_C_size, file);
    
    // Reallocate if needed for smaller matrix
    if (D_size * sizeof(float) > W_C_size * sizeof(float)) {
        temp = (float*)realloc(temp, D_size * sizeof(float));
    }
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_D, D_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), D_size, file);
    
    // Save Adam state
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_A_m, A_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), A_size, file);
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_A_v, A_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), A_size, file);
    
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_W_B_m, W_B_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), W_B_size, file);
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_W_B_v, W_B_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), W_B_size, file);
    
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_W_C_m, W_C_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), W_C_size, file);
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_W_C_v, W_C_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), W_C_size, file);
    
    // Reallocate if needed for D matrices
    if (D_size * sizeof(float) > W_C_size * sizeof(float)) {
        temp = (float*)realloc(temp, D_size * sizeof(float));
    }
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_D_m, D_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), D_size, file);
    CHECK_CUDA(cudaMemcpy(temp, ssm->d_D_v, D_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(temp, sizeof(float), D_size, file);
    
    free(temp);
    fclose(file);
    printf("Model saved to %s\n", filename);
}

SSM* load_ssm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, state_dim, output_dim, seq_len, stored_batch_size, t;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&t, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    ssm->t = t;
    
    // Calculate sizes
    int A_size = state_dim * state_dim;
    int W_B_size = state_dim * input_dim * input_dim;
    int W_C_size = output_dim * state_dim * state_dim;
    int D_size = output_dim * input_dim;
    
    // Allocate temporary host memory for largest matrix
    float* temp = (float*)malloc(W_C_size * sizeof(float));
    
    // Load matrices
    fread(temp, sizeof(float), A_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_A, temp, A_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(temp, sizeof(float), W_B_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B, temp, W_B_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(temp, sizeof(float), W_C_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_W_C, temp, W_C_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Reallocate if needed for smaller matrix
    if (D_size * sizeof(float) > W_C_size * sizeof(float)) {
        temp = (float*)realloc(temp, D_size * sizeof(float));
    }
    fread(temp, sizeof(float), D_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_D, temp, D_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(temp, sizeof(float), A_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_A_m, temp, A_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(temp, sizeof(float), A_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_A_v, temp, A_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(temp, sizeof(float), W_B_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B_m, temp, W_B_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(temp, sizeof(float), W_B_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_W_B_v, temp, W_B_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(temp, sizeof(float), W_C_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_W_C_m, temp, W_C_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(temp, sizeof(float), W_C_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_W_C_v, temp, W_C_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Reallocate if needed for D matrices
    if (D_size * sizeof(float) > W_C_size * sizeof(float)) {
        temp = (float*)realloc(temp, D_size * sizeof(float));
    }
    fread(temp, sizeof(float), D_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m, temp, D_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(temp, sizeof(float), D_size, file);
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v, temp, D_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(temp);
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return ssm;
}