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
    
    // Allocate state space matrices
    ssm->A = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate input-dependent projection parameters
    ssm->W_B = (float*)malloc(state_dim * input_dim * input_dim * sizeof(float));
    ssm->W_C = (float*)malloc(output_dim * state_dim * input_dim * sizeof(float));
    
    // Allocate gradients
    ssm->A_grad = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B_grad = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    ssm->W_B_grad = (float*)malloc(state_dim * input_dim * input_dim * sizeof(float));
    ssm->W_C_grad = (float*)malloc(output_dim * state_dim * input_dim * sizeof(float));
    
    // Allocate Adam buffers
    ssm->A_m = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->A_v = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->B_m = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->B_v = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->C_m = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->C_v = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->D_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->D_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->W_B_m = (float*)calloc(state_dim * input_dim * input_dim, sizeof(float));
    ssm->W_B_v = (float*)calloc(state_dim * input_dim * input_dim, sizeof(float));
    ssm->W_C_m = (float*)calloc(output_dim * state_dim * input_dim, sizeof(float));
    ssm->W_C_v = (float*)calloc(output_dim * state_dim * input_dim, sizeof(float));
    
    // Allocate helper arrays (time-major format)
    ssm->states = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->error = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->state_outputs = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    
    // Allocate temporary matrices for input-dependent projections
    ssm->B_t = (float*)malloc(batch_size * state_dim * input_dim * sizeof(float));
    ssm->C_t = (float*)malloc(batch_size * output_dim * state_dim * sizeof(float));
    
    // Initialize B, C, D matrices (kept for backward compatibility)
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
    
    // Initialize input-dependent projection parameters for selective SSM
    float scale_W_B = 0.1f / sqrtf(input_dim);
    float scale_W_C = 0.1f / sqrtf(input_dim);
    
    // Initialize W_B weights 
    for (int i = 0; i < state_dim * input_dim * input_dim; i++) {
        ssm->W_B[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W_B;
    }
    
    // Initialize W_C weights
    for (int i = 0; i < output_dim * state_dim * input_dim; i++) {
        ssm->W_C[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W_C;
    }
    
    // HiPPO-Leg inspired initialization for A matrix
    // Creates a lower triangular structure optimized for memory compression
    // and long-range dependency modeling
    // Note: A is already zero-initialized by calloc above
    
    // Phase 1: Create base lower triangular structure
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j <= i; j++) {
            if (i == j) {
                // Diagonal: negative values that increase in magnitude with index
                // This creates a structured forgetting pattern
                ssm->A[i * state_dim + j] = -0.01f - (i * 0.001f / state_dim);
            } else {
                // Off-diagonal: small positive values that decay with distance
                // This enables information flow between nearby state components
                float distance = i - j;
                ssm->A[i * state_dim + j] = 0.001f / (1.0f + distance * 0.1f);
            }
        }
    }
    
    // Phase 2: Apply Legendre polynomial scaling for optimal memory compression
    // This gives higher-order basis functions more importance
    float norm_factor = sqrtf(2.0f * state_dim + 1.0f);
    for (int i = 0; i < state_dim; i++) {
        float importance = sqrtf(2.0f * i + 1.0f);
        float normalized_importance = 1.0f + 0.1f * importance / norm_factor;
        
        // Scale entire row by importance factor
        for (int j = 0; j <= i; j++) {
            ssm->A[i * state_dim + j] *= normalized_importance;
        }
    }
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    free(ssm->A); free(ssm->B); free(ssm->C); free(ssm->D);
    free(ssm->W_B); free(ssm->W_C);
    free(ssm->A_grad); free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->W_B_grad); free(ssm->W_C_grad);
    free(ssm->A_m); free(ssm->A_v); free(ssm->B_m); free(ssm->B_v);
    free(ssm->C_m); free(ssm->C_v); free(ssm->D_m); free(ssm->D_v);
    free(ssm->W_B_m); free(ssm->W_B_v);
    free(ssm->W_C_m); free(ssm->W_C_v);
    free(ssm->states); free(ssm->predictions); free(ssm->error); free(ssm->state_error);
    free(ssm->state_outputs); free(ssm->B_t); free(ssm->C_t);
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
    
    // Compute input-dependent B_t = X_t * W_B (no bias)
    // B_t[b, s, i] = sum_j(X_t[b, j] * W_B[j, s, i])
    for (int b = 0; b < ssm->batch_size; b++) {
        for (int s = 0; s < ssm->state_dim; s++) {
            for (int i = 0; i < ssm->input_dim; i++) {
                int b_idx = b * ssm->state_dim * ssm->input_dim + s * ssm->input_dim + i;
                float* w_slice = ssm->W_B + s * ssm->input_dim * ssm->input_dim + i * ssm->input_dim;
                ssm->B_t[b_idx] = cblas_sdot(ssm->input_dim, X_t + b * ssm->input_dim, 1, w_slice, 1);
            }
        }
    }
    
    // Compute input-dependent C_t = X_t * W_C (no bias)
    // C_t[b, o, s] = sum_j(X_t[b, j] * W_C[j, o, s])
    for (int b = 0; b < ssm->batch_size; b++) {
        for (int o = 0; o < ssm->output_dim; o++) {
            for (int s = 0; s < ssm->state_dim; s++) {
                int c_idx = b * ssm->output_dim * ssm->state_dim + o * ssm->state_dim + s;
                float* w_slice = ssm->W_C + o * ssm->state_dim * ssm->input_dim + s * ssm->input_dim;
                ssm->C_t[c_idx] = cblas_sdot(ssm->input_dim, X_t + b * ssm->input_dim, 1, w_slice, 1);
            }
        }
    }
    
    // H_t = X_t B_t^T + H_{t-1} A^T
    // First compute X_t B_t^T using the input-dependent B_t
    memset(h_t, 0, ssm->batch_size * ssm->state_dim * sizeof(float));
    for (int b = 0; b < ssm->batch_size; b++) {
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    ssm->input_dim, ssm->state_dim,
                    1.0f, ssm->B_t + b * ssm->state_dim * ssm->input_dim, ssm->state_dim,
                    X_t + b * ssm->input_dim, 1,
                    1.0f, h_t + b * ssm->state_dim, 1);
    }
    
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
    
    // Y_t = O_t C_t^T + X_t D^T
    // First compute O_t C_t^T using the input-dependent C_t
    memset(y_t, 0, ssm->batch_size * ssm->output_dim * sizeof(float));
    for (int b = 0; b < ssm->batch_size; b++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    ssm->output_dim, ssm->state_dim,
                    1.0f, ssm->C_t + b * ssm->output_dim * ssm->state_dim, ssm->state_dim,
                    o_t + b * ssm->state_dim, 1,
                    1.0f, y_t + b * ssm->output_dim, 1);
    }
    
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
    memset(ssm->W_B_grad, 0, ssm->state_dim * ssm->input_dim * ssm->input_dim * sizeof(float));
    memset(ssm->W_C_grad, 0, ssm->output_dim * ssm->state_dim * ssm->input_dim * sizeof(float));
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
        
        // Recompute input-dependent matrices C_t for this timestep
        // C_t = X_t * W_C (no bias)
        // C_t[b, o, s] = sum_j(X_t[b, j] * W_C[j, o, s])
        for (int b = 0; b < ssm->batch_size; b++) {
            for (int o = 0; o < ssm->output_dim; o++) {
                for (int s = 0; s < ssm->state_dim; s++) {
                    int c_idx = b * ssm->output_dim * ssm->state_dim + o * ssm->state_dim + s;
                    float* w_slice = ssm->W_C + o * ssm->state_dim * ssm->input_dim + s * ssm->input_dim;
                    ssm->C_t[c_idx] = cblas_sdot(ssm->input_dim, X_t + b * ssm->input_dim, 1, w_slice, 1);
                }
            }
        }
        
        // ∂L/∂D += (∂L/∂Y_t)^T X_t (unchanged)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->output_dim, ssm->input_dim, ssm->batch_size,
                    1.0f, dy_t, ssm->output_dim,
                    X_t, ssm->input_dim,
                    1.0f, ssm->D_grad, ssm->input_dim);
        
        // Compute gradients for C projection parameters
        // ∂L/∂C_t from ∂L/∂Y_t (Y_t = O_t C_t^T + X_t D^T)
        // ∂L/∂C_t = O_t^T * ∂L/∂Y_t (for each batch element)
        for (int b = 0; b < ssm->batch_size; b++) {
            for (int o = 0; o < ssm->output_dim; o++) {
                for (int s = 0; s < ssm->state_dim; s++) {
                    float dc_t = dy_t[b * ssm->output_dim + o] * o_t[b * ssm->state_dim + s];
                    
                    // ∂L/∂W_C += X_t^T * ∂L/∂C_t
                    for (int i = 0; i < ssm->input_dim; i++) {
                        ssm->W_C_grad[o * ssm->state_dim * ssm->input_dim + s * ssm->input_dim + i] += 
                            X_t[b * ssm->input_dim + i] * dc_t;
                    }
                }
            }
        }
        
        // ∂L/∂O_t = (∂L/∂Y_t) * C_t (using input-dependent C_t)
        float* do_t = ssm->state_outputs + t * ssm->batch_size * ssm->state_dim; // reuse buffer
        memset(do_t, 0, ssm->batch_size * ssm->state_dim * sizeof(float));
        for (int b = 0; b < ssm->batch_size; b++) {
            for (int s = 0; s < ssm->state_dim; s++) {
                for (int o = 0; o < ssm->output_dim; o++) {
                    do_t[b * ssm->state_dim + s] += dy_t[b * ssm->output_dim + o] * 
                        ssm->C_t[b * ssm->output_dim * ssm->state_dim + o * ssm->state_dim + s];
                }
            }
        }
        
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
        
        // Recompute input-dependent B_t for this timestep
        // B_t = X_t * W_B (no bias)
        // B_t[b, s, i] = sum_j(X_t[b, j] * W_B[j, s, i])
        for (int b = 0; b < ssm->batch_size; b++) {
            for (int s = 0; s < ssm->state_dim; s++) {
                for (int i = 0; i < ssm->input_dim; i++) {
                    int b_idx = b * ssm->state_dim * ssm->input_dim + s * ssm->input_dim + i;
                    float* w_slice = ssm->W_B + s * ssm->input_dim * ssm->input_dim + i * ssm->input_dim;
                    ssm->B_t[b_idx] = cblas_sdot(ssm->input_dim, X_t + b * ssm->input_dim, 1, w_slice, 1);
                }
            }
        }
        
        // Compute gradients for B projection parameters
        // ∂L/∂B_t from ∂L/∂H_t (H_t = X_t B_t^T + H_{t-1} A^T)
        // ∂L/∂B_t = X_t^T * ∂L/∂H_t (for each batch element)
        for (int b = 0; b < ssm->batch_size; b++) {
            for (int s = 0; s < ssm->state_dim; s++) {
                for (int i = 0; i < ssm->input_dim; i++) {
                    float db_t = dh_t[b * ssm->state_dim + s] * X_t[b * ssm->input_dim + i];
                    
                    // ∂L/∂W_B += X_t^T * ∂L/∂B_t
                    for (int j = 0; j < ssm->input_dim; j++) {
                        ssm->W_B_grad[s * ssm->input_dim * ssm->input_dim + i * ssm->input_dim + j] += 
                            X_t[b * ssm->input_dim + j] * db_t;
                    }
                }
            }
        }
        
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
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update A
    for (int i = 0; i < ssm->state_dim * ssm->state_dim; i++) {
        float grad = ssm->A_grad[i] / ssm->batch_size;
        ssm->A_m[i] = ssm->beta1 * ssm->A_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->A_v[i] = ssm->beta2 * ssm->A_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->A_m[i] / (sqrtf(ssm->A_v[i]) + ssm->epsilon);
        ssm->A[i] = ssm->A[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
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
    
    // Update W_B
    for (int i = 0; i < ssm->state_dim * ssm->input_dim * ssm->input_dim; i++) {
        float grad = ssm->W_B_grad[i] / ssm->batch_size;
        ssm->W_B_m[i] = ssm->beta1 * ssm->W_B_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->W_B_v[i] = ssm->beta2 * ssm->W_B_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->W_B_m[i] / (sqrtf(ssm->W_B_v[i]) + ssm->epsilon);
        ssm->W_B[i] = ssm->W_B[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update W_C
    for (int i = 0; i < ssm->output_dim * ssm->state_dim * ssm->input_dim; i++) {
        float grad = ssm->W_C_grad[i] / ssm->batch_size;
        ssm->W_C_m[i] = ssm->beta1 * ssm->W_C_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->W_C_v[i] = ssm->beta2 * ssm->W_C_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->W_C_m[i] / (sqrtf(ssm->W_C_v[i]) + ssm->epsilon);
        ssm->W_C[i] = ssm->W_C[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
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
    
    // Save matrices
    fwrite(ssm->A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->W_B, sizeof(float), ssm->state_dim * ssm->input_dim * ssm->input_dim, file);
    fwrite(ssm->W_C, sizeof(float), ssm->output_dim * ssm->state_dim * ssm->input_dim, file);
    
    // Save Adam state
    fwrite(&ssm->t, sizeof(int), 1, file);
    fwrite(ssm->A_m, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->A_v, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B_m, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->B_v, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->W_B_m, sizeof(float), ssm->state_dim * ssm->input_dim * ssm->input_dim, file);
    fwrite(ssm->W_B_v, sizeof(float), ssm->state_dim * ssm->input_dim * ssm->input_dim, file);
    fwrite(ssm->W_C_m, sizeof(float), ssm->output_dim * ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->W_C_v, sizeof(float), ssm->output_dim * ssm->state_dim * ssm->input_dim, file);
    
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
    
    // Load matrices
    fread(ssm->A, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    fread(ssm->W_B, sizeof(float), state_dim * input_dim * input_dim, file);
    fread(ssm->W_C, sizeof(float), output_dim * state_dim * input_dim, file);
    
    // Load Adam state
    fread(&ssm->t, sizeof(int), 1, file);
    fread(ssm->A_m, sizeof(float), state_dim * state_dim, file);
    fread(ssm->A_v, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B_m, sizeof(float), state_dim * input_dim, file);
    fread(ssm->B_v, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C_m, sizeof(float), output_dim * state_dim, file);
    fread(ssm->C_v, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D_m, sizeof(float), output_dim * input_dim, file);
    fread(ssm->D_v, sizeof(float), output_dim * input_dim, file);
    fread(ssm->W_B_m, sizeof(float), state_dim * input_dim * input_dim, file);
    fread(ssm->W_B_v, sizeof(float), state_dim * input_dim * input_dim, file);
    fread(ssm->W_C_m, sizeof(float), output_dim * state_dim * input_dim, file);
    fread(ssm->W_C_v, sizeof(float), output_dim * state_dim * input_dim, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}
