#include "ssm.h"

// Padé approximation for matrix exponential exp(A)
static void matrix_exp_pade(float* result, const float* skew_matrix, int size, float* workspace, int* ipiv) {
    float* A2 = workspace;
    float* A4 = workspace + size * size;
    float* U = workspace + 2 * size * size;
    float* V = workspace + 3 * size * size;
    float* N = workspace + 4 * size * size;
    float* D = workspace + 5 * size * size;
    
    // A^2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0f, skew_matrix, size, skew_matrix, size, 0.0f, A2, size);
    
    // A^4 = A^2 * A^2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0f, A2, size, A2, size, 0.0f, A4, size);
    
    // Initialize V = 120*I + 120*A^2 + A^4
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int idx = i * size + j;
            V[idx] = 120.0f * A2[idx] + A4[idx];
            if (i == j) {
                V[idx] += 120.0f;  // 120*I
            }
        }
    }
    
    // U = A^4 + 60*A^2 + 120*I
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int idx = i * size + j;
            U[idx] = A4[idx] + 60.0f * A2[idx];
            if (i == j) {
                U[idx] += 120.0f;  // 120*I
            }
        }
    }
    
    // U = A * U
    memcpy(N, U, size * size * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0f, skew_matrix, size, N, size, 0.0f, U, size);
    
    // N = U + V, D = V - U
    for (int i = 0; i < size * size; i++) {
        N[i] = U[i] + V[i];
        D[i] = V[i] - U[i];
    }
    
    // Solve D * result = N
    memcpy(result, N, size * size * sizeof(float));
    LAPACKE_sgetrf(LAPACK_ROW_MAJOR, size, size, D, size, ipiv);
    LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', size, size, D, size, ipiv, result, size);
}

// Construct skew-symmetric matrix from parameters
static void construct_skew_matrix(float* skew_matrix, const float* params, int size) {
    memset(skew_matrix, 0, size * size * sizeof(float));
    
    int param_idx = 0;
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            float val = params[param_idx++];
            skew_matrix[i * size + j] = val;
            skew_matrix[j * size + i] = -val;
        }
    }
}

// Compute A matrix from block-diagonal structure
static void compute_A_from_blocks(SSM* ssm) {
    memset(ssm->A, 0, ssm->state_dim * ssm->state_dim * sizeof(float));
    
    float* skew_block = ssm->workspace;
    float* exp_block = ssm->workspace + ssm->block_size * ssm->block_size;
    float* pade_work = ssm->workspace + 2 * ssm->block_size * ssm->block_size;
    
    for (int block = 0; block < ssm->num_blocks; block++) {
        int block_start = block * ssm->block_size;
        const float* block_params = ssm->A_skew + block * (ssm->block_size * (ssm->block_size - 1) / 2);
        
        // Construct skew-symmetric matrix for this block
        construct_skew_matrix(skew_block, block_params, ssm->block_size);
        
        // Compute matrix exponential
        matrix_exp_pade(exp_block, skew_block, ssm->block_size, pade_work, ssm->ipiv);
        
        // Copy to appropriate block in A
        for (int i = 0; i < ssm->block_size; i++) {
            for (int j = 0; j < ssm->block_size; j++) {
                int A_i = block_start + i;
                int A_j = block_start + j;
                if (A_i < ssm->state_dim && A_j < ssm->state_dim) {
                    ssm->A[A_i * ssm->state_dim + A_j] = exp_block[i * ssm->block_size + j];
                }
            }
        }
    }
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
    
    // Block configuration
    ssm->block_size = 16;
    ssm->num_blocks = (state_dim + ssm->block_size - 1) / ssm->block_size;

    // Allocate state space matrices
    ssm->A = (float*)calloc(ssm->state_dim * ssm->state_dim, sizeof(float));
    ssm->B = (float*)malloc(ssm->state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * ssm->state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate block-diagonal parameters
    ssm->A_skew = (float*)malloc((ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2) * sizeof(float));
    
    // Allocate gradients
    ssm->A_grad = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    ssm->B_grad = (float*)malloc(ssm->state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * ssm->state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    ssm->A_skew_grad = (float*)malloc((ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2) * sizeof(float));
    
    // Allocate Adam buffers
    ssm->B_m = (float*)calloc(ssm->state_dim * input_dim, sizeof(float));
    ssm->B_v = (float*)calloc(ssm->state_dim * input_dim, sizeof(float));
    ssm->C_m = (float*)calloc(output_dim * ssm->state_dim, sizeof(float));
    ssm->C_v = (float*)calloc(output_dim * ssm->state_dim, sizeof(float));
    ssm->D_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->D_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->A_skew_m = (float*)calloc((ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2), sizeof(float));
    ssm->A_skew_v = (float*)calloc((ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2), sizeof(float));
    
    // Allocate helper arrays (time-major format)
    ssm->states = (float*)malloc(seq_len * batch_size * ssm->state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->error = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(seq_len * batch_size * ssm->state_dim * sizeof(float));
    ssm->state_outputs = (float*)malloc(seq_len * batch_size * ssm->state_dim * sizeof(float));
    
    // Allocate unified workspace
    ssm->workspace = (float*)malloc(12 * ssm->block_size * ssm->block_size * sizeof(float));
    ssm->ipiv = (int*)malloc(ssm->block_size * sizeof(int));
    
    // Initialize B, C, D matrices
    float scale_B = 0.5f / sqrtf(input_dim);
    float scale_C = 0.5f / sqrtf(ssm->state_dim);
    float scale_D = 0.1f / sqrtf(input_dim);
    
    for (int i = 0; i < ssm->state_dim * input_dim; i++) {
        ssm->B[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_B;
    }
    
    for (int i = 0; i < output_dim * ssm->state_dim; i++) {
        ssm->C[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->D[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_D;
    }
    
    // Initialize skew-symmetric parameters
    float skew_scale = 0.1f;
    for (int i = 0; i < (ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2); i++) {
        ssm->A_skew[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * skew_scale;
    }
    
    // Compute initial A matrix from blocks
    compute_A_from_blocks(ssm);
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    free(ssm->A); free(ssm->B); free(ssm->C); free(ssm->D);
    free(ssm->A_skew);
    free(ssm->A_grad); free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->A_skew_grad);
    free(ssm->B_m); free(ssm->B_v); free(ssm->C_m); free(ssm->C_v);
    free(ssm->D_m); free(ssm->D_v); free(ssm->A_skew_m); free(ssm->A_skew_v);
    free(ssm->states); free(ssm->predictions); free(ssm->error); free(ssm->state_error);
    free(ssm->state_outputs);
    free(ssm->workspace); free(ssm->ipiv);
    free(ssm);
}

// Reset hidden states to zero
void reset_state_ssm(SSM* ssm) {
    memset(ssm->states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
}

// Forward pass
void forward_pass_ssm(SSM* ssm, float* X_t, int timestep) {
    // Get pointers to current timestep state
    float* h_prev = (timestep > 0) ? ssm->states + (timestep - 1) * ssm->batch_size * ssm->state_dim : NULL;
    float* h_t = ssm->states + timestep * ssm->batch_size * ssm->state_dim;
    float* o_t = ssm->state_outputs + timestep * ssm->batch_size * ssm->state_dim;
    float* y_t = ssm->predictions + timestep * ssm->batch_size * ssm->output_dim;
        
    // H_t = X_t B^T + H_{t-1} A^T
    // H_t = X_t B^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim, ssm->input_dim,
                1.0f, X_t, ssm->input_dim,
                ssm->B, ssm->input_dim,
                0.0f, h_t, ssm->state_dim);
    
    // H_t += H_{t-1} A^T
    if (timestep > 0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->state_dim, ssm->state_dim,
                    1.0f, h_prev, ssm->state_dim,
                    ssm->A, ssm->state_dim,
                    1.0f, h_t, ssm->state_dim);
    }
    
    // O_t = H_t σ(H_t)
    for (int i = 0; i < ssm->batch_size * ssm->state_dim; i++) {
        float h = h_t[i];
        o_t[i] = h / (1.0f + expf(-h));
    }
    
    // Y_t = O_t C^T + X_t D^T
    // Y_t = O_t C^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->output_dim, ssm->state_dim,
                1.0f, o_t, ssm->state_dim,
                ssm->C, ssm->state_dim,
                0.0f, y_t, ssm->output_dim);
    
    // Y_t += X_t D^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->output_dim, ssm->input_dim,
                1.0f, X_t, ssm->input_dim,
                ssm->D, ssm->input_dim,
                1.0f, y_t, ssm->output_dim);
}

// Calculate loss
float calculate_loss_ssm(SSM* ssm, float* y) {
    float loss = 0.0f;
    int total_size = ssm->seq_len * ssm->batch_size * ssm->output_dim;
    
    for (int i = 0; i < total_size; i++) {
        ssm->error[i] = ssm->predictions[i] - y[i];
        loss += ssm->error[i] * ssm->error[i];
    }
    
    return loss / total_size;
}

// Zero gradients
void zero_gradients_ssm(SSM* ssm) {
    memset(ssm->A_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float));
    memset(ssm->B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float));
    memset(ssm->C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float));
    memset(ssm->D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float));
    memset(ssm->A_skew_grad, 0, (ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2) * sizeof(float));
}

// Backward pass
void backward_pass_ssm(SSM* ssm, float* X) {
    // Clear state errors
    memset(ssm->state_error, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    
    for (int t = ssm->seq_len - 1; t >= 0; t--) {
        float* X_t = X + t * ssm->batch_size * ssm->input_dim;
        float* h_t = ssm->states + t * ssm->batch_size * ssm->state_dim;
        float* o_t = ssm->state_outputs + t * ssm->batch_size * ssm->state_dim;
        float* dy_t = ssm->error + t * ssm->batch_size * ssm->output_dim;
        float* dh_t = ssm->state_error + t * ssm->batch_size * ssm->state_dim;
        
        // ∂L/∂C += (∂L/∂Y_t)^T O_t
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->output_dim, ssm->state_dim, ssm->batch_size,
                    1.0f, dy_t, ssm->output_dim,
                    o_t, ssm->state_dim,
                    1.0f, ssm->C_grad, ssm->state_dim);
        
        // ∂L/∂D += (∂L/∂Y_t)^T X_t
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->output_dim, ssm->input_dim, ssm->batch_size,
                    1.0f, dy_t, ssm->output_dim,
                    X_t, ssm->input_dim,
                    1.0f, ssm->D_grad, ssm->input_dim);
        
        // ∂L/∂O_t = (∂L/∂Y_t)C
        float* do_t = ssm->state_outputs + t * ssm->batch_size * ssm->state_dim; // reuse buffer
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    ssm->batch_size, ssm->state_dim, ssm->output_dim,
                    1.0f, dy_t, ssm->output_dim,
                    ssm->C, ssm->state_dim,
                    0.0f, do_t, ssm->state_dim);
        
        // ∂L/∂H_t = ∂L/∂O_t ⊙ [σ(H_t) + H_t σ(H_t)(1-σ(H_t))]
        for (int i = 0; i < ssm->batch_size * ssm->state_dim; i++) {
            float h = h_t[i];
            float sigmoid = 1.0f / (1.0f + expf(-h));
            dh_t[i] = do_t[i] * sigmoid * (1.0f + h * (1.0f - sigmoid));
        }
        
        // ∂L/∂H_t += (∂L/∂H_{t+1})A
        if (t < ssm->seq_len - 1) {
            float* dh_next = ssm->state_error + (t+1) * ssm->batch_size * ssm->state_dim;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        ssm->batch_size, ssm->state_dim, ssm->state_dim,
                        1.0f, dh_next, ssm->state_dim,
                        ssm->A, ssm->state_dim,
                        1.0f, dh_t, ssm->state_dim);
        }
        
        // ∂L/∂B += (∂L/∂H_t)^T X_t
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->state_dim, ssm->input_dim, ssm->batch_size,
                    1.0f, dh_t, ssm->state_dim,
                    X_t, ssm->input_dim,
                    1.0f, ssm->B_grad, ssm->input_dim);
        
        // ∂L/∂A += (∂L/∂H_t)^T H_{t-1}
        if (t > 0) {
            float* h_prev = ssm->states + (t-1) * ssm->batch_size * ssm->state_dim;
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        ssm->state_dim, ssm->state_dim, ssm->batch_size,
                        1.0f, dh_t, ssm->state_dim,
                        h_prev, ssm->state_dim,
                        1.0f, ssm->A_grad, ssm->state_dim);
        }
    }
    
    // Compute gradients with respect to skew parameters
    memset(ssm->A_skew_grad, 0, (ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2) * sizeof(float));
    
    float* skew_block = ssm->workspace;
    float* exp_block = ssm->workspace + ssm->block_size * ssm->block_size;
    float* pade_work = ssm->workspace + 2 * ssm->block_size * ssm->block_size;
    float* grad_block = ssm->workspace + 8 * ssm->block_size * ssm->block_size;
    
    for (int block = 0; block < ssm->num_blocks; block++) {
        int block_start = block * ssm->block_size;
        const float* block_params = ssm->A_skew + block * (ssm->block_size * (ssm->block_size - 1) / 2);
        float* block_grad_params = ssm->A_skew_grad + block * (ssm->block_size * (ssm->block_size - 1) / 2);
        
        // Extract gradient block from A_grad
        for (int i = 0; i < ssm->block_size; i++) {
            for (int j = 0; j < ssm->block_size; j++) {
                int A_i = block_start + i;
                int A_j = block_start + j;
                if (A_i < ssm->state_dim && A_j < ssm->state_dim) {
                    grad_block[i * ssm->block_size + j] = ssm->A_grad[A_i * ssm->state_dim + A_j];
                } else {
                    grad_block[i * ssm->block_size + j] = 0.0f;
                }
            }
        }
        
        // Construct skew-symmetric matrix and compute exponential for this block
        construct_skew_matrix(skew_block, block_params, ssm->block_size);
        matrix_exp_pade(exp_block, skew_block, ssm->block_size, pade_work, ssm->ipiv);
        
        // Analytical gradient computation using Fréchet derivative
        // For matrix exponential, the Fréchet derivative in direction E is:
        // dexp(S)[E] = ∫₀¹ exp(tS) E exp((1-t)S) dt
        // 
        // For small ||S||, we can use the approximation:
        // dexp(S)[E] ≈ E + (SE + ES)/2 + higher order terms
        //
        // Since we're dealing with skew-symmetric matrices (small norms by design),
        // we use: dexp(S)/ds_ij ≈ (E_ij exp(S) + exp(S) E_ij) / 2
        // where E_ij has 1 at (i,j), -1 at (j,i) for skew symmetry
        
        float* temp1 = ssm->workspace + 9 * ssm->block_size * ssm->block_size;
        float* temp2 = ssm->workspace + 10 * ssm->block_size * ssm->block_size;
        float* E_ij = ssm->workspace + 11 * ssm->block_size * ssm->block_size;
        
        int param_idx = 0;
        for (int i = 0; i < ssm->block_size; i++) {
            for (int j = i + 1; j < ssm->block_size; j++) {
                // Create elementary skew-symmetric matrix E_ij
                memset(E_ij, 0, ssm->block_size * ssm->block_size * sizeof(float));
                E_ij[i * ssm->block_size + j] = 1.0f;
                E_ij[j * ssm->block_size + i] = -1.0f;
                
                // Compute improved approximation for small skew matrices:
                // dexp(S)/ds_ij ≈ E_ij + (S E_ij + E_ij S)/2 + (S E_ij + E_ij S) exp(S)/2
                
                // temp1 = S * E_ij
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            ssm->block_size, ssm->block_size, ssm->block_size,
                            1.0f, skew_block, ssm->block_size, E_ij, ssm->block_size,
                            0.0f, temp1, ssm->block_size);
                
                // temp2 = E_ij * S
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            ssm->block_size, ssm->block_size, ssm->block_size,
                            1.0f, E_ij, ssm->block_size, skew_block, ssm->block_size,
                            0.0f, temp2, ssm->block_size);
                
                // For the derivative computation, we use a more accurate formula:
                // For orthogonal matrices exp(S) where S is skew-symmetric,
                // dexp(S)/ds_ij can be computed as:
                // (exp(S) * E_ij + E_ij * exp(S)) / 2
                
                // temp1 = exp(S) * E_ij
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            ssm->block_size, ssm->block_size, ssm->block_size,
                            1.0f, exp_block, ssm->block_size, E_ij, ssm->block_size,
                            0.0f, temp1, ssm->block_size);
                
                // temp2 = E_ij * exp(S)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            ssm->block_size, ssm->block_size, ssm->block_size,
                            1.0f, E_ij, ssm->block_size, exp_block, ssm->block_size,
                            0.0f, temp2, ssm->block_size);
                
                // Chain rule: ∂L/∂s_ij = trace(∂L/∂A * ∂A/∂s_ij)
                float grad_sum = 0.0f;
                for (int k = 0; k < ssm->block_size * ssm->block_size; k++) {
                    // Average of the two terms for symmetry
                    float derivative = 0.5f * (temp1[k] + temp2[k]);
                    grad_sum += grad_block[k] * derivative;
                }
                
                block_grad_params[param_idx] = grad_sum;
                param_idx++;
            }
        }
    }
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update A_skew parameters
    for (int i = 0; i < (ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2); i++) {
        float grad = ssm->A_skew_grad[i] / ssm->batch_size;
        ssm->A_skew_m[i] = ssm->beta1 * ssm->A_skew_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->A_skew_v[i] = ssm->beta2 * ssm->A_skew_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->A_skew_m[i] / (sqrtf(ssm->A_skew_v[i]) + ssm->epsilon);
        ssm->A_skew[i] = ssm->A_skew[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update B
    for (int i = 0; i < ssm->state_dim * ssm->input_dim; i++) {
        float grad = ssm->B_grad[i] / ssm->batch_size;
        ssm->B_m[i] = ssm->beta1 * ssm->B_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->B_v[i] = ssm->beta2 * ssm->B_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->B_m[i] / (sqrtf(ssm->B_v[i]) + ssm->epsilon);
        ssm->B[i] = ssm->B[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update C
    for (int i = 0; i < ssm->output_dim * ssm->state_dim; i++) {
        float grad = ssm->C_grad[i] / ssm->batch_size;
        ssm->C_m[i] = ssm->beta1 * ssm->C_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->C_v[i] = ssm->beta2 * ssm->C_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->C_m[i] / (sqrtf(ssm->C_v[i]) + ssm->epsilon);
        ssm->C[i] = ssm->C[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update D
    for (int i = 0; i < ssm->output_dim * ssm->input_dim; i++) {
        float grad = ssm->D_grad[i] / ssm->batch_size;
        ssm->D_m[i] = ssm->beta1 * ssm->D_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->D_v[i] = ssm->beta2 * ssm->D_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->D_m[i] / (sqrtf(ssm->D_v[i]) + ssm->epsilon);
        ssm->D[i] = ssm->D[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Recompute A matrix from updated skew parameters
    compute_A_from_blocks(ssm);
}

// Save model
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
    fwrite(&ssm->num_blocks, sizeof(int), 1, file);
    fwrite(&ssm->block_size, sizeof(int), 1, file);
    
    // Save matrices
    fwrite(ssm->A_skew, sizeof(float), (ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2), file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Save Adam state
    fwrite(&ssm->t, sizeof(int), 1, file);
    fwrite(ssm->A_skew_m, sizeof(float), (ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2), file);
    fwrite(ssm->A_skew_v, sizeof(float), (ssm->num_blocks * ssm->block_size * (ssm->block_size - 1) / 2), file);
    fwrite(ssm->B_m, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->B_v, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
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
    int num_blocks, block_size;
    
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&num_blocks, sizeof(int), 1, file);
    fread(&block_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Load matrices
    fread(ssm->A_skew, sizeof(float), (num_blocks * block_size * (block_size - 1) / 2), file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    
    // Load Adam state
    fread(&ssm->t, sizeof(int), 1, file);
    fread(ssm->A_skew_m, sizeof(float), (num_blocks * block_size * (block_size - 1) / 2), file);
    fread(ssm->A_skew_v, sizeof(float), (num_blocks * block_size * (block_size - 1) / 2), file);
    fread(ssm->B_m, sizeof(float), state_dim * input_dim, file);
    fread(ssm->B_v, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C_m, sizeof(float), output_dim * state_dim, file);
    fread(ssm->C_v, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D_m, sizeof(float), output_dim * input_dim, file);
    fread(ssm->D_v, sizeof(float), output_dim * input_dim, file);
    
    // Recompute A matrix from loaded skew parameters
    compute_A_from_blocks(ssm);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}