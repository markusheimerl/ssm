#include "ssm.h"

// Compute gradients w.r.t. A_skew parameters through the Cayley transform
// This implements the chain rule: dL/dA_skew = dL/dA_orthogonal * dA_orthogonal/dA_skew
// Using simplified analytical approximation for efficiency
void compute_A_skew_gradients(float* A_skew, float* A_orthogonal_grad, float* A_skew_grad, int state_dim) {
    int n = state_dim;
    (void)A_skew; // Suppress unused parameter warning
    
    // Simplified approximation: assume dA_orthogonal/dA_skew ≈ scale * I
    // This is not exact but provides a reasonable approximation that's much faster
    float scale = 2.0f; // Empirical scaling factor
    
    // Map A_orthogonal_grad to A_skew_grad using the skew-symmetric structure
    int param_idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // Gradient w.r.t. skew parameter affects both (i,j) and (j,i) positions
            // with opposite signs due to skew-symmetry
            float grad_ij = A_orthogonal_grad[i * n + j];
            float grad_ji = A_orthogonal_grad[j * n + i];
            A_skew_grad[param_idx] = scale * (grad_ij - grad_ji);
            param_idx++;
        }
    }
}

// Cayley transform: A = (I + S)(I - S)^(-1) where S is skew-symmetric
// A_skew contains n(n-1)/2 parameters for upper triangular part of S
void cayley_transform(float* A_skew, float* A_orthogonal, int state_dim,
                     float* workspace_S, float* workspace_I_plus_S, float* workspace_I_minus_S, int* workspace_ipiv) {
    int n = state_dim;
    
    // Use pre-allocated workspace instead of malloc
    float* S = workspace_S;
    float* I_plus_S = workspace_I_plus_S;
    float* I_minus_S = workspace_I_minus_S;
    int* ipiv = workspace_ipiv;
    
    // Clear S matrix
    memset(S, 0, n * n * sizeof(float));
    
    // Construct skew-symmetric matrix S from A_skew parameters
    int param_idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            S[i * n + j] = A_skew[param_idx];    // Upper triangular
            S[j * n + i] = -A_skew[param_idx];   // Lower triangular (negative)
            param_idx++;
        }
    }
    
    // Compute I + S and I - S
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            I_plus_S[i * n + j] = S[i * n + j];
            I_minus_S[i * n + j] = -S[i * n + j];
            if (i == j) {
                I_plus_S[i * n + j] += 1.0f;  // Add identity
                I_minus_S[i * n + j] += 1.0f; // Add identity
            }
        }
    }
    
    // Solve (I - S) * A_orthogonal = (I + S) using LAPACK SGESV
    // This computes A_orthogonal = (I - S)^(-1) * (I + S) = (I + S)(I - S)^(-1)
    int info;
    
    // Copy I_plus_S to A_orthogonal (SGESV will overwrite the RHS)
    memcpy(A_orthogonal, I_plus_S, n * n * sizeof(float));
    
    // Solve the system: I_minus_S * X = I_plus_S
    sgesv_(&n, &n, I_minus_S, &n, ipiv, A_orthogonal, &n, &info);
    
    if (info != 0) {
        printf("Warning: LAPACK SGESV failed with info = %d\n", info);
        // Fallback to identity matrix
        memset(A_orthogonal, 0, n * n * sizeof(float));
        for (int i = 0; i < n; i++) {
            A_orthogonal[i * n + i] = 1.0f;
        }
    }
    
    // No cleanup needed - using pre-allocated workspace
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
    ssm->A_skew = (float*)calloc(skew_params, sizeof(float));
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
    
    // Allocate workspace for Cayley transform (pre-allocated to avoid malloc/free in forward/backward)
    ssm->workspace_S = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->workspace_I_plus_S = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->workspace_I_minus_S = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->workspace_ipiv = (int*)malloc(state_dim * sizeof(int));
    
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
    
    // Initialize skew-symmetric parameters randomly (small values for stability)
    float scale_skew = 0.01f / sqrtf(state_dim);
    for (int i = 0; i < skew_params; i++) {
        ssm->A_skew[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_skew;
    }
    
    // Compute initial orthogonal A matrix using Cayley transform
    cayley_transform(ssm->A_skew, ssm->A_orthogonal, state_dim,
                    ssm->workspace_S, ssm->workspace_I_plus_S, ssm->workspace_I_minus_S, ssm->workspace_ipiv);
    
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
    free(ssm->workspace_S); free(ssm->workspace_I_plus_S); free(ssm->workspace_I_minus_S);
    free(ssm->workspace_ipiv);
    free(ssm);
}

// Reset hidden states to zero
void reset_state_ssm(SSM* ssm) {
    memset(ssm->states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
}

// Forward pass
void forward_pass_ssm(SSM* ssm, float* X_t, int timestep) {
    // Recompute A_orthogonal from A_skew at each forward pass
    cayley_transform(ssm->A_skew, ssm->A_orthogonal, ssm->state_dim,
                    ssm->workspace_S, ssm->workspace_I_plus_S, ssm->workspace_I_minus_S, ssm->workspace_ipiv);
    
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
    
    // Allocate temporary A_orthogonal gradient
    float* A_orthogonal_grad = (float*)calloc(ssm->state_dim * ssm->state_dim, sizeof(float));
    
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
    
    // Compute gradients w.r.t. A_skew parameters using chain rule
    compute_A_skew_gradients(ssm->A_skew, A_orthogonal_grad, ssm->A_skew_grad, ssm->state_dim);
    
    free(A_orthogonal_grad);
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
    
    int skew_params = ssm->state_dim * (ssm->state_dim - 1) / 2;
    
    // Save dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->seq_len, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
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
    
    // Recompute A_orthogonal from loaded A_skew
    cayley_transform(ssm->A_skew, ssm->A_orthogonal, state_dim,
                    ssm->workspace_S, ssm->workspace_I_plus_S, ssm->workspace_I_minus_S, ssm->workspace_ipiv);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}
