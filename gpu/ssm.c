#include "ssm.h"

// CUDA kernel for applying Givens rotation to matrix at positions (i,j)
__global__ void apply_givens_rotation_kernel_ssm(float* matrix, int n, int i, int j, float cos_theta, float sin_theta) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        float a_ik = matrix[i * n + k];
        float a_jk = matrix[j * n + k];
        matrix[i * n + k] = cos_theta * a_ik - sin_theta * a_jk;
        matrix[j * n + k] = sin_theta * a_ik + cos_theta * a_jk;
    }
}

// CUDA kernel for copying matrix
__global__ void copy_matrix_kernel_ssm(float* dest, float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] = src[idx];
    }
}

// Optimized kernel that builds orthogonal matrix entirely on GPU
__global__ void build_orthogonal_optimized_kernel_ssm(float* A_orthogonal, float* rotation_angles, int state_dim) {
    extern __shared__ float sdata[];
    
    int total_size = state_dim * state_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize to identity
    if (idx < total_size) {
        int i = idx / state_dim;
        int j = idx % state_dim;
        A_orthogonal[idx] = (i == j) ? 1.0f : 0.0f;
    }
    
    __syncthreads();
    
    // Apply rotations - coordinate across all threads
    int angle_idx = 0;
    for (int rot_i = 1; rot_i < state_dim; rot_i++) {
        for (int rot_j = 0; rot_j < rot_i; rot_j++) {
            __syncthreads(); // Ensure previous rotation is complete
            
            // Load angle and compute cos/sin
            float cos_theta, sin_theta;
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                float theta = rotation_angles[angle_idx];
                cos_theta = cosf(theta);
                sin_theta = sinf(theta);
                sdata[0] = cos_theta;
                sdata[1] = sin_theta;
            }
            
            __syncthreads();
            
            // All threads read shared cos/sin values
            cos_theta = sdata[0];
            sin_theta = sdata[1];
            
            // Each thread handles one column
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k < state_dim) {
                float a_ik = A_orthogonal[rot_i * state_dim + k];
                float a_jk = A_orthogonal[rot_j * state_dim + k];
                A_orthogonal[rot_i * state_dim + k] = cos_theta * a_ik - sin_theta * a_jk;
                A_orthogonal[rot_j * state_dim + k] = sin_theta * a_ik + cos_theta * a_jk;
            }
            
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                angle_idx++;
            }
            __syncthreads();
        }
    }
}

// Build orthogonal matrix from rotation angles (host function) 
void build_orthogonal_from_angles(SSM* ssm) {
    int state_dim = ssm->state_dim;
    int block_size = 256;
    int num_blocks = (state_dim + block_size - 1) / block_size;
    
    // Use optimized kernel that runs entirely on GPU with shared memory for cos/sin
    build_orthogonal_optimized_kernel_ssm<<<num_blocks, block_size, 2 * sizeof(float)>>>(
        ssm->d_A_orthogonal, ssm->d_rotation_angles, state_dim);
    CHECK_CUDA(cudaDeviceSynchronize());
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
    
    // Calculate number of rotation angles: n(n-1)/2
    int num_angles = state_dim * (state_dim - 1) / 2;
    
    // Allocate host memory for initialization
    float* A = (float*)calloc(state_dim * state_dim, sizeof(float)); // Keep for compatibility
    float* rotation_angles = (float*)malloc(num_angles * sizeof(float));
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
    
    // Initialize rotation angles randomly in [-π/4, π/4] for stability
    for (int i = 0; i < num_angles; i++) {
        rotation_angles[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * (M_PI / 4.0f);
    }
    
    // Allocate device memory for matrices
    CHECK_CUDA(cudaMalloc(&ssm->d_A, state_dim * state_dim * sizeof(float))); // Keep for compatibility
    CHECK_CUDA(cudaMalloc(&ssm->d_rotation_angles, num_angles * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_orthogonal, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, state_dim * state_dim * sizeof(float))); // Keep for compatibility
    CHECK_CUDA(cudaMalloc(&ssm->d_rotation_angles_grad, num_angles * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ssm->d_rotation_angles_m, num_angles * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_rotation_angles_v, num_angles * sizeof(float)));
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
    
    // Allocate temporary GPU buffers for gradient computation (to avoid malloc/free in backward pass)
    CHECK_CUDA(cudaMalloc(&ssm->d_A_temp, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_orthogonal_grad, state_dim * state_dim * sizeof(float)));
    
    // Copy initialized matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice)); // Keep for compatibility
    CHECK_CUDA(cudaMemcpy(ssm->d_rotation_angles, rotation_angles, num_angles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ssm->d_rotation_angles_m, 0, num_angles * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_rotation_angles_v, 0, num_angles * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    
    // Build initial orthogonal A matrix from rotation angles
    build_orthogonal_from_angles(ssm);
    
    // Free host memory
    free(A);
    free(rotation_angles);
    free(B);
    free(C);
    free(D);
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    // Free device memory
    cudaFree(ssm->d_A); cudaFree(ssm->d_rotation_angles); cudaFree(ssm->d_A_orthogonal);
    cudaFree(ssm->d_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);
    cudaFree(ssm->d_A_grad); cudaFree(ssm->d_rotation_angles_grad);
    cudaFree(ssm->d_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_rotation_angles_m); cudaFree(ssm->d_rotation_angles_v);
    cudaFree(ssm->d_B_m); cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m); cudaFree(ssm->d_C_v); cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);
    cudaFree(ssm->d_states); cudaFree(ssm->d_predictions); cudaFree(ssm->d_error); 
    cudaFree(ssm->d_state_error); cudaFree(ssm->d_state_outputs);
    cudaFree(ssm->d_A_temp); cudaFree(ssm->d_A_orthogonal_grad);  // Free temporary buffers
    
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
    
    // Build orthogonal A matrix from current rotation angles
    build_orthogonal_from_angles(ssm);
    
    // Get pointers to current timestep state
    float* d_h_prev = (timestep > 0) ? ssm->d_states + (timestep - 1) * ssm->batch_size * ssm->state_dim : NULL;
    float* d_h_t = ssm->d_states + timestep * ssm->batch_size * ssm->state_dim;
    float* d_o_t = ssm->d_state_outputs + timestep * ssm->batch_size * ssm->state_dim;
    float* d_y_t = ssm->d_predictions + timestep * ssm->batch_size * ssm->output_dim;
    
    // H_t = X_t B^T + H_{t-1} A_orthogonal^T
    // H_t = X_t B^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim,
                            &alpha, ssm->d_B, ssm->input_dim,
                            d_X_t, ssm->input_dim,
                            &beta, d_h_t, ssm->state_dim));
    
    // H_t += H_{t-1} A_orthogonal^T
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
    int num_angles = ssm->state_dim * (ssm->state_dim - 1) / 2;
    CHECK_CUDA(cudaMemset(ssm->d_rotation_angles_grad, 0, num_angles * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float)));
}

// Optimized kernel for computing all rotation gradients in parallel
__global__ void compute_all_rotation_gradients_kernel_ssm(
    float* rotation_grad, float* A_grad, float* A_temp, float* rotation_angles, 
    int state_dim) {
    
    extern __shared__ float sdata[];
    int num_angles = state_dim * (state_dim - 1) / 2;
    int angle_idx = blockIdx.x;
    
    if (angle_idx >= num_angles) return;
    
    // Compute which rotation this angle corresponds to
    int rot_i = 1, rot_j = 0;
    int temp_idx = 0;
    for (int i = 1; i < state_dim; i++) {
        for (int j = 0; j < i; j++) {
            if (temp_idx == angle_idx) {
                rot_i = i;
                rot_j = j;
                goto found_rotation;
            }
            temp_idx++;
        }
    }
    
    found_rotation:
    
    float theta = rotation_angles[angle_idx];
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    
    // Compute gradient contribution for this angle across all columns in parallel
    float grad_theta_total = 0.0f;
    
    for (int k = threadIdx.x; k < state_dim; k += blockDim.x) {
        float grad_i_k = A_grad[rot_i * state_dim + k];
        float grad_j_k = A_grad[rot_j * state_dim + k];
        float a_i_k = A_temp[rot_i * state_dim + k];
        float a_j_k = A_temp[rot_j * state_dim + k];
        
        // ∂/∂θ (cos(θ) * a_ik - sin(θ) * a_jk) = -sin(θ) * a_ik - cos(θ) * a_jk
        // ∂/∂θ (sin(θ) * a_ik + cos(θ) * a_jk) = cos(θ) * a_ik - sin(θ) * a_jk
        float grad_theta_contrib = grad_i_k * (-sin_theta * a_i_k - cos_theta * a_j_k) +
                                   grad_j_k * (cos_theta * a_i_k - sin_theta * a_j_k);
        grad_theta_total += grad_theta_contrib;
    }
    
    // Reduce across threads in this block
    sdata[threadIdx.x] = grad_theta_total;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        rotation_grad[angle_idx] = sdata[0];
    }
}

// Optimized kernel for applying transpose rotations without host-device transfers
__global__ void apply_all_transpose_rotations_kernel_ssm(
    float* A_grad, float* A_temp, float* rotation_angles, int state_dim) {
    
    int num_angles = state_dim * (state_dim - 1) / 2;
    
    // Process rotations in reverse order
    for (int angle_idx = num_angles - 1; angle_idx >= 0; angle_idx--) {
        // Compute which rotation this angle corresponds to
        int rot_i = 1, rot_j = 0;
        int temp_idx = 0;
        for (int i = 1; i < state_dim; i++) {
            for (int j = 0; j < i; j++) {
                if (temp_idx == angle_idx) {
                    rot_i = i;
                    rot_j = j;
                    goto found_rotation;
                }
                temp_idx++;
            }
        }
        
        found_rotation:
        
        float theta = rotation_angles[angle_idx];
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        
        __syncthreads(); // Ensure all threads have the same rotation parameters
        
        // Apply transpose rotation to gradients
        int k = threadIdx.x + blockIdx.x * blockDim.x;
        if (k < state_dim) {
            float grad_i_k = A_grad[rot_i * state_dim + k];
            float grad_j_k = A_grad[rot_j * state_dim + k];
            
            A_grad[rot_i * state_dim + k] = cos_theta * grad_i_k + sin_theta * grad_j_k;
            A_grad[rot_j * state_dim + k] = -sin_theta * grad_i_k + cos_theta * grad_j_k;
            
            // Apply to temporary matrix too
            float a_i_k = A_temp[rot_i * state_dim + k];
            float a_j_k = A_temp[rot_j * state_dim + k];
            A_temp[rot_i * state_dim + k] = cos_theta * a_i_k - sin_theta * a_j_k;
            A_temp[rot_j * state_dim + k] = sin_theta * a_i_k + cos_theta * a_j_k;
        }
        
        __syncthreads(); // Ensure rotation is complete before next iteration
    }
}

// Compute gradients of rotation angles from gradients of orthogonal matrix (GPU version)
void compute_rotation_gradients_gpu(SSM* ssm, float* d_A_orthogonal_grad) {
    int n = ssm->state_dim;
    int num_angles = n * (n - 1) / 2;
    
    // Copy current orthogonal matrix to temporary buffer
    int total_size = n * n;
    dim3 copy_block(256);
    dim3 copy_grid((total_size + copy_block.x - 1) / copy_block.x);
    copy_matrix_kernel_ssm<<<copy_grid, copy_block>>>(ssm->d_A_temp, ssm->d_A_orthogonal, total_size);
    
    // Initialize rotation angle gradients to zero
    CHECK_CUDA(cudaMemset(ssm->d_rotation_angles_grad, 0, num_angles * sizeof(float)));
    
    // Compute all rotation gradients in parallel - one block per angle
    int block_size = 256;
    compute_all_rotation_gradients_kernel_ssm<<<num_angles, block_size, block_size * sizeof(float)>>>(
        ssm->d_rotation_angles_grad, d_A_orthogonal_grad, ssm->d_A_temp, ssm->d_rotation_angles, n);
    
    // Apply all transpose rotations in sequence but without host-device transfers
    dim3 transpose_block(256);
    dim3 transpose_grid((n + transpose_block.x - 1) / transpose_block.x);
    apply_all_transpose_rotations_kernel_ssm<<<transpose_grid, transpose_block>>>(
        d_A_orthogonal_grad, ssm->d_A_temp, ssm->d_rotation_angles, n);
    
    CHECK_CUDA(cudaDeviceSynchronize());
}
// Backward pass
void backward_pass_ssm(SSM* ssm, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Clear state errors
    CHECK_CUDA(cudaMemset(ssm->d_state_error, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float)));
    
    // Use pre-allocated gradient buffer instead of malloc
    float* d_A_orthogonal_grad = ssm->d_A_orthogonal_grad;
    CHECK_CUDA(cudaMemset(d_A_orthogonal_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float)));
    
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
        
        // ∂L/∂H_t += (∂L/∂H_{t+1})A_orthogonal
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
        
        // ∂L/∂A_orthogonal += (∂L/∂H_t)^T H_{t-1}
        if (t > 0) {
            float* d_h_prev = ssm->d_states + (t-1) * ssm->batch_size * ssm->state_dim;
            CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                    ssm->state_dim, ssm->state_dim, ssm->batch_size,
                                    &alpha, d_h_prev, ssm->state_dim,
                                    d_dh_t, ssm->state_dim,
                                    &beta_add, d_A_orthogonal_grad, ssm->state_dim));
        }
    }
    
    // Compute gradients w.r.t. rotation angles using chain rule
    compute_rotation_gradients_gpu(ssm, d_A_orthogonal_grad);
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
    int num_angles = ssm->state_dim * (ssm->state_dim - 1) / 2;
    
    // Update rotation_angles
    int angles_blocks = (num_angles + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<angles_blocks, block_size>>>(
        ssm->d_rotation_angles, ssm->d_rotation_angles_grad, ssm->d_rotation_angles_m, ssm->d_rotation_angles_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, num_angles, ssm->batch_size
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
    int num_angles = ssm->state_dim * (ssm->state_dim - 1) / 2;
    
    // Allocate temporary host memory
    float* rotation_angles = (float*)malloc(num_angles * sizeof(float));
    float* B = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* C = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(rotation_angles, ssm->d_rotation_angles, num_angles * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B, ssm->d_B, ssm->state_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C, ssm->d_C, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D, ssm->d_D, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(rotation_angles); free(B); free(C); free(D);
        return;
    }
    
    // Save dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->seq_len, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
    // Save rotation angles and matrices
    fwrite(rotation_angles, sizeof(float), num_angles, file);
    fwrite(B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    fwrite(&ssm->t, sizeof(int), 1, file);
    
    // Save Adam state
    float* rotation_angles_m = (float*)malloc(num_angles * sizeof(float));
    float* rotation_angles_v = (float*)malloc(num_angles * sizeof(float));
    float* B_m = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* B_v = (float*)malloc(ssm->state_dim * ssm->input_dim * sizeof(float));
    float* C_m = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* C_v = (float*)malloc(ssm->output_dim * ssm->state_dim * sizeof(float));
    float* D_m = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    float* D_v = (float*)malloc(ssm->output_dim * ssm->input_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(rotation_angles_m, ssm->d_rotation_angles_m, num_angles * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(rotation_angles_v, ssm->d_rotation_angles_v, num_angles * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_m, ssm->d_B_m, ssm->state_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_v, ssm->d_B_v, ssm->state_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_m, ssm->d_C_m, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_v, ssm->d_C_v, ssm->output_dim * ssm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_m, ssm->d_D_m, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_v, ssm->d_D_v, ssm->output_dim * ssm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(rotation_angles_m, sizeof(float), num_angles, file);
    fwrite(rotation_angles_v, sizeof(float), num_angles, file);
    fwrite(B_m, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(B_v, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Free temporary host memory
    free(rotation_angles); free(B); free(C); free(D);
    free(rotation_angles_m); free(rotation_angles_v); free(B_m); free(B_v);
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
    int num_angles = state_dim * (state_dim - 1) / 2;
    
    // Initialize model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Allocate temporary host memory
    float* rotation_angles = (float*)malloc(num_angles * sizeof(float));
    float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Load rotation angles and matrices
    fread(rotation_angles, sizeof(float), num_angles, file);
    fread(B, sizeof(float), state_dim * input_dim, file);
    fread(C, sizeof(float), output_dim * state_dim, file);
    fread(D, sizeof(float), output_dim * input_dim, file);
    
    fread(&ssm->t, sizeof(int), 1, file);
    
    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(ssm->d_rotation_angles, rotation_angles, num_angles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    float* rotation_angles_m = (float*)malloc(num_angles * sizeof(float));
    float* rotation_angles_v = (float*)malloc(num_angles * sizeof(float));
    float* B_m = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* B_v = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C_m = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* C_v = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D_m = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* D_v = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    fread(rotation_angles_m, sizeof(float), num_angles, file);
    fread(rotation_angles_v, sizeof(float), num_angles, file);
    fread(B_m, sizeof(float), state_dim * input_dim, file);
    fread(B_v, sizeof(float), state_dim * input_dim, file);
    fread(C_m, sizeof(float), output_dim * state_dim, file);
    fread(C_v, sizeof(float), output_dim * state_dim, file);
    fread(D_m, sizeof(float), output_dim * input_dim, file);
    fread(D_v, sizeof(float), output_dim * input_dim, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(ssm->d_rotation_angles_m, rotation_angles_m, num_angles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_rotation_angles_v, rotation_angles_v, num_angles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_m, B_m, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_v, B_v, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_m, C_m, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_v, C_v, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m, D_m, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v, D_v, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Build orthogonal A matrix from loaded rotation angles
    build_orthogonal_from_angles(ssm);
    
    // Free temporary host memory
    free(rotation_angles); free(B); free(C); free(D);
    free(rotation_angles_m); free(rotation_angles_v); free(B_m); free(B_v);
    free(C_m); free(C_v); free(D_m); free(D_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}
