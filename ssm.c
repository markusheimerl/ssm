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
    
    // Allocate gradients
    ssm->A_grad = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B_grad = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate Adam buffers
    ssm->A_m = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->A_v = (float*)calloc(state_dim * state_dim, sizeof(float));
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
    free(ssm->A_grad); free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->A_m); free(ssm->A_v); free(ssm->B_m); free(ssm->B_v);
    free(ssm->C_m); free(ssm->C_v); free(ssm->D_m); free(ssm->D_v);
    free(ssm->states); free(ssm->predictions); free(ssm->error); free(ssm->state_error);
    free(ssm->state_outputs);
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
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}

// Parallel forward pass using OpenMP parallelization across batch dimension
void forward_pass_ssm_parallel(SSM* ssm, float* X) {
    // Clear states first
    memset(ssm->states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    
    int seq_len = ssm->seq_len;
    int batch_size = ssm->batch_size;
    int state_dim = ssm->state_dim;
    int input_dim = ssm->input_dim;
    int output_dim = ssm->output_dim;
    
    // Process each batch element in parallel
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        // For each batch element, process time sequentially (due to recurrence)
        for (int t = 0; t < seq_len; t++) {
            float* X_t = X + t * batch_size * input_dim + b * input_dim;
            float* h_t = ssm->states + t * batch_size * state_dim + b * state_dim;
            float* o_t = ssm->state_outputs + t * batch_size * state_dim + b * state_dim;
            float* y_t = ssm->predictions + t * batch_size * output_dim + b * output_dim;
            
            // H_t = X_t B^T - single batch element version
            // Use sgemm with batch_size=1 to match original implementation
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, state_dim, input_dim,
                        1.0f, X_t, input_dim,
                        ssm->B, input_dim,
                        0.0f, h_t, state_dim);
            
            // H_t += H_{t-1} A^T (if t > 0)
            if (t > 0) {
                float* h_prev = ssm->states + (t-1) * batch_size * state_dim + b * state_dim;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                           1, state_dim, state_dim,
                           1.0f, h_prev, state_dim,
                           ssm->A, state_dim,
                           1.0f, h_t, state_dim);
            }
            
            // O_t = H_t * σ(H_t) - apply Swish activation element-wise
            for (int i = 0; i < state_dim; i++) {
                float h = h_t[i];
                o_t[i] = h / (1.0f + expf(-h));
            }
            
            // Y_t = O_t * C^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                       1, output_dim, state_dim,
                       1.0f, o_t, state_dim,
                       ssm->C, state_dim,
                       0.0f, y_t, output_dim);
            
            // Y_t += X_t * D^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                       1, output_dim, input_dim,
                       1.0f, X_t, input_dim,
                       ssm->D, input_dim,
                       1.0f, y_t, output_dim);
        }
    }
}

// Helper function to multiply two affine transformation matrices
// Each transformation represents: h_new = A * h_old + b
// Composition: h_new = A2 * (A1 * h_old + b1) + b2 = (A2 * A1) * h_old + (A2 * b1 + b2)
void compose_affine_transforms(float* A1, float* b1, float* A2, float* b2,
                              float* result_A, float* result_b, int state_dim) {
    // result_A = A2 * A1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                state_dim, state_dim, state_dim,
                1.0f, A2, state_dim,
                A1, state_dim,
                0.0f, result_A, state_dim);
    
    // result_b = A2 * b1 + b2
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                state_dim, state_dim,
                1.0f, A2, state_dim,
                b1, 1,
                0.0f, result_b, 1);
    
    cblas_saxpy(state_dim, 1.0f, b2, 1, result_b, 1);
}

// True Blelloch parallel scan implementation for SSM
void forward_pass_ssm_blelloch_scan(SSM* ssm, float* X) {
    // Clear states first
    memset(ssm->states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    
    int seq_len = ssm->seq_len;
    int batch_size = ssm->batch_size;
    int state_dim = ssm->state_dim;
    int input_dim = ssm->input_dim;
    int output_dim = ssm->output_dim;
    
    // Process each batch element in parallel
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        // Allocate temporary matrices for scan operation
        // Each timestep is represented as an affine transformation: h_t = A_t * h_{t-1} + b_t
        float* A_matrices = (float*)malloc(seq_len * state_dim * state_dim * sizeof(float));
        float* b_vectors = (float*)malloc(seq_len * state_dim * sizeof(float));
        float* temp_A = (float*)malloc(state_dim * state_dim * sizeof(float));
        float* temp_b = (float*)malloc(state_dim * sizeof(float));
        
        // Step 1: Initialize transformations for each timestep
        for (int t = 0; t < seq_len; t++) {
            float* X_t = X + t * batch_size * input_dim + b * input_dim;
            float* A_t = A_matrices + t * state_dim * state_dim;
            float* b_t = b_vectors + t * state_dim;
            
            // A_t = A^T (the state transition matrix transposed)
            for (int i = 0; i < state_dim; i++) {
                for (int j = 0; j < state_dim; j++) {
                    A_t[i * state_dim + j] = ssm->A[j * state_dim + i];
                }
            }
            
            // b_t = X_t * B^T (the input contribution)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                       1, state_dim, input_dim,
                       1.0f, X_t, input_dim,
                       ssm->B, input_dim,
                       0.0f, b_t, state_dim);
        }
        
        // Step 2: Blelloch scan (up-sweep and down-sweep)
        int n = seq_len;
        int levels = 0;
        int temp_n = n;
        while (temp_n > 1) {
            temp_n = (temp_n + 1) / 2;
            levels++;
        }
        
        // Up-sweep phase: build the tree of partial results
        for (int level = 0; level < levels; level++) {
            int step = 1 << (level + 1);
            for (int i = step - 1; i < n; i += step) {
                int left = i - (1 << level);
                if (left >= 0) {
                    float* A_left = A_matrices + left * state_dim * state_dim;
                    float* b_left = b_vectors + left * state_dim;
                    float* A_right = A_matrices + i * state_dim * state_dim;
                    float* b_right = b_vectors + i * state_dim;
                    
                    // Compose: right = right ∘ left
                    compose_affine_transforms(A_left, b_left, A_right, b_right,
                                            temp_A, temp_b, state_dim);
                    
                    memcpy(A_right, temp_A, state_dim * state_dim * sizeof(float));
                    memcpy(b_right, temp_b, state_dim * sizeof(float));
                }
            }
        }
        
        // Clear the last element for down-sweep
        if (n > 0) {
            float* A_last = A_matrices + (n-1) * state_dim * state_dim;
            float* b_last = b_vectors + (n-1) * state_dim;
            
            // Set to identity transformation
            memset(A_last, 0, state_dim * state_dim * sizeof(float));
            for (int i = 0; i < state_dim; i++) {
                A_last[i * state_dim + i] = 1.0f;
            }
            memset(b_last, 0, state_dim * sizeof(float));
        }
        
        // Down-sweep phase: propagate partial results
        for (int level = levels - 1; level >= 0; level--) {
            int step = 1 << (level + 1);
            for (int i = step - 1; i < n; i += step) {
                int right = i + (1 << level);
                if (right < n) {
                    float* A_curr = A_matrices + i * state_dim * state_dim;
                    float* b_curr = b_vectors + i * state_dim;
                    float* A_right = A_matrices + right * state_dim * state_dim;
                    float* b_right = b_vectors + right * state_dim;
                    
                    // Save current values
                    memcpy(temp_A, A_right, state_dim * state_dim * sizeof(float));
                    memcpy(temp_b, b_right, state_dim * sizeof(float));
                    
                    // right = curr ∘ right
                    compose_affine_transforms(temp_A, temp_b, A_curr, b_curr,
                                            A_right, b_right, state_dim);
                    
                    // curr = saved right
                    memcpy(A_curr, temp_A, state_dim * state_dim * sizeof(float));
                    memcpy(b_curr, temp_b, state_dim * sizeof(float));
                }
            }
        }
        
        // Step 3: Apply transformations to compute final states
        // Since we start with h_{-1} = 0, h_t = b_t for all t
        for (int t = 0; t < seq_len; t++) {
            float* h_t = ssm->states + t * batch_size * state_dim + b * state_dim;
            float* b_t = b_vectors + t * state_dim;
            memcpy(h_t, b_t, state_dim * sizeof(float));
        }
        
        // Step 4: Compute outputs for this batch element
        for (int t = 0; t < seq_len; t++) {
            float* X_t = X + t * batch_size * input_dim + b * input_dim;
            float* h_t = ssm->states + t * batch_size * state_dim + b * state_dim;
            float* o_t = ssm->state_outputs + t * batch_size * state_dim + b * state_dim;
            float* y_t = ssm->predictions + t * batch_size * output_dim + b * output_dim;
            
            // O_t = H_t * σ(H_t) - apply Swish activation element-wise
            for (int i = 0; i < state_dim; i++) {
                float h = h_t[i];
                o_t[i] = h / (1.0f + expf(-h));
            }
            
            // Y_t = O_t * C^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                       1, output_dim, state_dim,
                       1.0f, o_t, state_dim,
                       ssm->C, state_dim,
                       0.0f, y_t, output_dim);
            
            // Y_t += X_t * D^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                       1, output_dim, input_dim,
                       1.0f, X_t, input_dim,
                       ssm->D, input_dim,
                       1.0f, y_t, output_dim);
        }
        
        free(A_matrices);
        free(b_vectors);
        free(temp_A);
        free(temp_b);
    }
}
