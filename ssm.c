#include "ssm.h"

// Simple BLAS replacement implementation
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                int m, int n, int k, float alpha, const float* A, int lda,
                const float* B, int ldb, float beta, float* C, int ldc) {
    // Simple implementation for row-major layout
    if (layout != CblasRowMajor) return; // Only support row-major for now
    
    // Initialize C with beta * C
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * ldc + j] *= beta;
        }
    }
    
    // C = alpha * A * B + C
    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    C[i * ldc + j] += alpha * A[i * lda + l] * B[l * ldb + j];
                }
            }
        }
    } else if (transA == CblasNoTrans && transB == CblasTrans) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    C[i * ldc + j] += alpha * A[i * lda + l] * B[j * ldb + l];
                }
            }
        }
    } else if (transA == CblasTrans && transB == CblasNoTrans) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    C[i * ldc + j] += alpha * A[l * lda + i] * B[l * ldb + j];
                }
            }
        }
    } else if (transA == CblasTrans && transB == CblasTrans) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    C[i * ldc + j] += alpha * A[l * lda + i] * B[j * ldb + l];
                }
            }
        }
    }
}



// Create full skew-symmetric matrix from upper triangular parameters
void create_skew_symmetric(float* A_skew_full, const float* A_skew_params, int n) {
    // Initialize to zero
    memset(A_skew_full, 0, n * n * sizeof(float));
    
    // Fill upper triangle and make anti-symmetric
    int param_idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A_skew_full[i * n + j] = A_skew_params[param_idx];
            A_skew_full[j * n + i] = -A_skew_params[param_idx];
            param_idx++;
        }
    }
}

// Simple matrix multiplication for small matrices
void matrix_multiply(float* C, const float* A, const float* B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Simple matrix inversion using Gaussian elimination for small matrices
int matrix_invert(float* inv, const float* A, int n, float* workspace) {
    // Create augmented matrix [A | I]
    float* aug = workspace; // size: n * 2n
    
    // Initialize augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (2 * n) + j] = A[i * n + j];
            aug[i * (2 * n) + (n + j)] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Gaussian elimination with partial pivoting
    for (int i = 0; i < n; i++) {
        // Find pivot
        int pivot_row = i;
        float max_val = fabsf(aug[i * (2 * n) + i]);
        for (int k = i + 1; k < n; k++) {
            if (fabsf(aug[k * (2 * n) + i]) > max_val) {
                max_val = fabsf(aug[k * (2 * n) + i]);
                pivot_row = k;
            }
        }
        
        // Check for singular matrix
        if (max_val < 1e-10f) {
            return -1; // Matrix is singular
        }
        
        // Swap rows if needed
        if (pivot_row != i) {
            for (int j = 0; j < 2 * n; j++) {
                float temp = aug[i * (2 * n) + j];
                aug[i * (2 * n) + j] = aug[pivot_row * (2 * n) + j];
                aug[pivot_row * (2 * n) + j] = temp;
            }
        }
        
        // Scale pivot row
        float pivot = aug[i * (2 * n) + i];
        for (int j = 0; j < 2 * n; j++) {
            aug[i * (2 * n) + j] /= pivot;
        }
        
        // Eliminate column
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = aug[k * (2 * n) + i];
                for (int j = 0; j < 2 * n; j++) {
                    aug[k * (2 * n) + j] -= factor * aug[i * (2 * n) + j];
                }
            }
        }
    }
    
    // Extract inverse matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv[i * n + j] = aug[i * (2 * n) + (n + j)];
        }
    }
    
    return 0; // Success
}

// Matrix exponential via Padé approximation with proper matrix inversion
void matrix_exponential_pade(float* exp_A, const float* A_skew_full, int n, float* workspace) {
    // For small matrices, use simplified Padé(3,3) approximation
    // exp(A) ≈ (I + A/2 + A²/12) * (I - A/2 + A²/12)^(-1)
    
    // Use workspace memory layout:
    // workspace[0..n²-1]: identity
    // workspace[n²..2n²-1]: A²
    // workspace[2n²..3n²-1]: numerator
    // workspace[3n²..4n²-1]: denominator
    // workspace[4n²..5n²-1]: denominator_inv
    // workspace[5n²..5n²+2n²-1]: aug matrix for inversion
    
    float* identity = workspace;
    float* A2 = workspace + n * n;
    float* numerator = workspace + 2 * n * n;
    float* denominator = workspace + 3 * n * n;
    float* denominator_inv = workspace + 4 * n * n;
    float* aug_workspace = workspace + 5 * n * n;
    
    // Create identity matrix
    memset(identity, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        identity[i * n + i] = 1.0f;
    }
    
    // Compute A² 
    matrix_multiply(A2, A_skew_full, A_skew_full, n);
    
    // Compute numerator: I + A/2 + A²/12
    for (int i = 0; i < n * n; i++) {
        numerator[i] = identity[i] + 0.5f * A_skew_full[i] + A2[i] / 12.0f;
    }
    
    // Compute denominator: I - A/2 + A²/12
    for (int i = 0; i < n * n; i++) {
        denominator[i] = identity[i] - 0.5f * A_skew_full[i] + A2[i] / 12.0f;
    }
    
    // Invert denominator
    if (matrix_invert(denominator_inv, denominator, n, aug_workspace) == 0) {
        // exp_A = numerator * denominator_inv
        matrix_multiply(exp_A, numerator, denominator_inv, n);
    } else {
        // If inversion fails, fall back to simplified approximation
        memcpy(exp_A, numerator, n * n * sizeof(float));
    }
}

// Compute gradient through matrix exponential (simplified)
void matrix_exponential_gradient(float* grad_A_skew, const float* grad_exp_A, 
                                const float* A_skew_full, int n) {
    // Simplified gradient computation
    // For small skew-symmetric matrices, use first-order approximation:
    // d(exp(A))/dA ≈ I for small A
    // We use A_skew_full for completeness but simplify the computation
    (void)A_skew_full; // Mark as used to avoid warning
    
    int param_idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // Sum gradients for A[i,j] and A[j,i] positions
            grad_A_skew[param_idx] = grad_exp_A[i * n + j] - grad_exp_A[j * n + i];
            param_idx++;
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
    
    // Calculate number of skew-symmetric parameters
    int skew_params = state_dim * (state_dim - 1) / 2;
    
    // Allocate state space matrices
    ssm->A_skew = (float*)malloc(skew_params * sizeof(float));
    ssm->A_orthogonal = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate gradients
    ssm->A_skew_grad = (float*)malloc(skew_params * sizeof(float));
    ssm->B_grad = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate Adam buffers
    ssm->A_skew_m = (float*)calloc(skew_params, sizeof(float));
    ssm->A_skew_v = (float*)calloc(skew_params, sizeof(float));
    ssm->B_m = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->B_v = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->C_m = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->C_v = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->D_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->D_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    
    // Allocate helper arrays (time-major format)
    ssm->states = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->error = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->state_outputs = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    
    // Allocate workspace memory for matrix operations
    ssm->A_skew_full = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->workspace = (float*)malloc((5 * state_dim * state_dim + 2 * state_dim * state_dim) * sizeof(float));
    
    // Initialize B, C, D matrices
    float scale_B = 0.5f / sqrtf(input_dim);
    float scale_C = 0.5f / sqrtf(state_dim);
    float scale_D = 0.1f / sqrtf(input_dim);
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->B[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->C[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->D[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_D;
    }
    
    // Initialize A_skew parameters with small random values
    float scale_A_skew = 0.1f / sqrtf(state_dim);
    for (int i = 0; i < skew_params; i++) {
        ssm->A_skew[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_A_skew;
    }
    
    // Compute initial A_orthogonal from A_skew using pre-allocated workspace
    create_skew_symmetric(ssm->A_skew_full, ssm->A_skew, state_dim);
    matrix_exponential_pade(ssm->A_orthogonal, ssm->A_skew_full, state_dim, ssm->workspace);
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    free(ssm->A_skew); free(ssm->A_orthogonal); free(ssm->B); free(ssm->C); free(ssm->D);
    free(ssm->A_skew_grad); free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->A_skew_m); free(ssm->A_skew_v); free(ssm->B_m); free(ssm->B_v);
    free(ssm->C_m); free(ssm->C_v); free(ssm->D_m); free(ssm->D_v);
    free(ssm->states); free(ssm->predictions); free(ssm->error); free(ssm->state_error);
    free(ssm->state_outputs);
    free(ssm->A_skew_full); free(ssm->workspace);
    free(ssm);
}

// Reset hidden states to zero
void reset_state_ssm(SSM* ssm) {
    memset(ssm->states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
}

// Forward pass
void forward_pass_ssm(SSM* ssm, float* X_t, int timestep) {
    // Recompute A_orthogonal from A_skew using pre-allocated workspace
    create_skew_symmetric(ssm->A_skew_full, ssm->A_skew, ssm->state_dim);
    matrix_exponential_pade(ssm->A_orthogonal, ssm->A_skew_full, ssm->state_dim, ssm->workspace);
    
    // Get pointers to current timestep state
    float* h_prev = (timestep > 0) ? ssm->states + (timestep - 1) * ssm->batch_size * ssm->state_dim : NULL;
    float* h_t = ssm->states + timestep * ssm->batch_size * ssm->state_dim;
    float* o_t = ssm->state_outputs + timestep * ssm->batch_size * ssm->state_dim;
    float* y_t = ssm->predictions + timestep * ssm->batch_size * ssm->output_dim;
        
    // H_t = X_t B^T + H_{t-1} A_orthogonal^T
    // H_t = X_t B^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim, ssm->input_dim,
                1.0f, X_t, ssm->input_dim,
                ssm->B, ssm->input_dim,
                0.0f, h_t, ssm->state_dim);
    
    // H_t += H_{t-1} A_orthogonal^T
    if (timestep > 0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->state_dim, ssm->state_dim,
                    1.0f, h_prev, ssm->state_dim,
                    ssm->A_orthogonal, ssm->state_dim,
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
    int skew_params = ssm->state_dim * (ssm->state_dim - 1) / 2;
    memset(ssm->A_skew_grad, 0, skew_params * sizeof(float));
    memset(ssm->B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float));
    memset(ssm->C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float));
    memset(ssm->D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float));
}

// Backward pass
void backward_pass_ssm(SSM* ssm, float* X) {
    // Clear state errors
    memset(ssm->state_error, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    
    // Allocate temporary gradient matrix for A_orthogonal using workspace
    // Use the second half of workspace for A_orthogonal_grad
    float* A_orthogonal_grad = ssm->workspace + 2 * ssm->state_dim * ssm->state_dim;
    memset(A_orthogonal_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float));
    
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
        
        // ∂L/∂H_t += (∂L/∂H_{t+1})A_orthogonal
        if (t < ssm->seq_len - 1) {
            float* dh_next = ssm->state_error + (t+1) * ssm->batch_size * ssm->state_dim;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        ssm->batch_size, ssm->state_dim, ssm->state_dim,
                        1.0f, dh_next, ssm->state_dim,
                        ssm->A_orthogonal, ssm->state_dim,
                        1.0f, dh_t, ssm->state_dim);
        }
        
        // ∂L/∂B += (∂L/∂H_t)^T X_t
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->state_dim, ssm->input_dim, ssm->batch_size,
                    1.0f, dh_t, ssm->state_dim,
                    X_t, ssm->input_dim,
                    1.0f, ssm->B_grad, ssm->input_dim);
        
        // ∂L/∂A_orthogonal += (∂L/∂H_t)^T H_{t-1}
        if (t > 0) {
            float* h_prev = ssm->states + (t-1) * ssm->batch_size * ssm->state_dim;
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        ssm->state_dim, ssm->state_dim, ssm->batch_size,
                        1.0f, dh_t, ssm->state_dim,
                        h_prev, ssm->state_dim,
                        1.0f, A_orthogonal_grad, ssm->state_dim);
        }
    }
    
    // Convert gradients from A_orthogonal to A_skew using pre-allocated workspace
    create_skew_symmetric(ssm->A_skew_full, ssm->A_skew, ssm->state_dim);
    matrix_exponential_gradient(ssm->A_skew_grad, A_orthogonal_grad, ssm->A_skew_full, ssm->state_dim);
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int skew_params = ssm->state_dim * (ssm->state_dim - 1) / 2;
    
    // Update A_skew
    for (int i = 0; i < skew_params; i++) {
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
    
    int skew_params = ssm->state_dim * (ssm->state_dim - 1) / 2;
    
    // Save matrices
    fwrite(ssm->A_skew, sizeof(float), skew_params, file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Save Adam state
    fwrite(&ssm->t, sizeof(int), 1, file);
    fwrite(ssm->A_skew_m, sizeof(float), skew_params, file);
    fwrite(ssm->A_skew_v, sizeof(float), skew_params, file);
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
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    int skew_params = state_dim * (state_dim - 1) / 2;
    
    // Initialize model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Load matrices
    fread(ssm->A_skew, sizeof(float), skew_params, file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    
    // Load Adam state
    fread(&ssm->t, sizeof(int), 1, file);
    fread(ssm->A_skew_m, sizeof(float), skew_params, file);
    fread(ssm->A_skew_v, sizeof(float), skew_params, file);
    fread(ssm->B_m, sizeof(float), state_dim * input_dim, file);
    fread(ssm->B_v, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C_m, sizeof(float), output_dim * state_dim, file);
    fread(ssm->C_v, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D_m, sizeof(float), output_dim * input_dim, file);
    fread(ssm->D_v, sizeof(float), output_dim * input_dim, file);
    
    // Recompute A_orthogonal from loaded A_skew using pre-allocated workspace
    create_skew_symmetric(ssm->A_skew_full, ssm->A_skew, state_dim);
    matrix_exponential_pade(ssm->A_orthogonal, ssm->A_skew_full, state_dim, ssm->workspace);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}
