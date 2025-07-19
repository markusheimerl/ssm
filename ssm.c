#include "ssm.h"

// Initialize the state space model
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    
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

// Free the state space model
void free_ssm(SSM* ssm) {
    free(ssm->A); free(ssm->B); free(ssm->C); free(ssm->D);
    free(ssm->A_grad); free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->A_m); free(ssm->A_v); free(ssm->B_m); free(ssm->B_v);
    free(ssm->C_m); free(ssm->C_v); free(ssm->D_m); free(ssm->D_v);
    free(ssm->states); free(ssm->predictions); free(ssm->error); free(ssm->state_error);
    free(ssm);
}

// Forward pass
void forward_pass_ssm(SSM* ssm, float* X) {
    // Clear states
    memset(ssm->states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    
    for (int t = 0; t < ssm->seq_len; t++) {
        float* X_t = X + t * ssm->batch_size * ssm->input_dim;
        float* h_t = ssm->states + t * ssm->batch_size * ssm->state_dim;
        
        // H_t = X_t B^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->state_dim, ssm->input_dim,
                    1.0f, X_t, ssm->input_dim,
                    ssm->B, ssm->input_dim,
                    0.0f, h_t, ssm->state_dim);
        
        // H_t += H_{t-1} A^T
        if (t > 0) {
            float* h_prev = ssm->states + (t-1) * ssm->batch_size * ssm->state_dim;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ssm->batch_size, ssm->state_dim, ssm->state_dim,
                        1.0f, h_prev, ssm->state_dim,
                        ssm->A, ssm->state_dim,
                        1.0f, h_t, ssm->state_dim);
        }
        
        // Y_t = H_t C^T + X_t D^T (using hidden state directly)
        float* y_t = ssm->predictions + t * ssm->batch_size * ssm->output_dim;
        
        // Y_t = H_t C^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->output_dim, ssm->state_dim,
                    1.0f, h_t, ssm->state_dim,
                    ssm->C, ssm->state_dim,
                    0.0f, y_t, ssm->output_dim);
        
        // Y_t += X_t D^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->output_dim, ssm->input_dim,
                    1.0f, X_t, ssm->input_dim,
                    ssm->D, ssm->input_dim,
                    1.0f, y_t, ssm->output_dim);
    }
}

// Calculate loss
float calculate_loss_ssm(SSM* ssm, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < ssm->seq_len * ssm->batch_size * ssm->output_dim; i++) {
        float diff = ssm->predictions[i] - y[i];
        ssm->error[i] = diff;
        loss += diff * diff;
    }
    return loss / (ssm->seq_len * ssm->batch_size * ssm->output_dim);
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
        float* dy_t = ssm->error + t * ssm->batch_size * ssm->output_dim;
        float* dh_t = ssm->state_error + t * ssm->batch_size * ssm->state_dim;
        
        // ∂L/∂C += (∂L/∂Y_t)^T H_t (using hidden state directly)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->output_dim, ssm->state_dim, ssm->batch_size,
                    1.0f, dy_t, ssm->output_dim,
                    h_t, ssm->state_dim,
                    1.0f, ssm->C_grad, ssm->state_dim);
        
        // ∂L/∂D += (∂L/∂Y_t)^T X_t
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->output_dim, ssm->input_dim, ssm->batch_size,
                    1.0f, dy_t, ssm->output_dim,
                    X_t, ssm->input_dim,
                    1.0f, ssm->D_grad, ssm->input_dim);
        
        // ∂L/∂H_t = (∂L/∂Y_t)C (compute gradients directly to hidden state)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    ssm->batch_size, ssm->state_dim, ssm->output_dim,
                    1.0f, dy_t, ssm->output_dim,
                    ssm->C, ssm->state_dim,
                    0.0f, dh_t, ssm->state_dim);
        
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

// Update weights using Adam optimizer
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    const float beta1_corrected = 1.0f - powf(ssm->beta1, ssm->t);
    const float beta2_corrected = 1.0f - powf(ssm->beta2, ssm->t);
    
    // Update A matrix
    for (int i = 0; i < ssm->state_dim * ssm->state_dim; i++) {
        // Add weight decay
        ssm->A_grad[i] += ssm->weight_decay * ssm->A[i];
        
        // Update biased first moment estimate
        ssm->A_m[i] = ssm->beta1 * ssm->A_m[i] + (1.0f - ssm->beta1) * ssm->A_grad[i];
        
        // Update biased second raw moment estimate
        ssm->A_v[i] = ssm->beta2 * ssm->A_v[i] + (1.0f - ssm->beta2) * ssm->A_grad[i] * ssm->A_grad[i];
        
        // Compute bias-corrected moment estimates
        float m_hat = ssm->A_m[i] / beta1_corrected;
        float v_hat = ssm->A_v[i] / beta2_corrected;
        
        // Update parameters
        ssm->A[i] -= learning_rate * m_hat / (sqrtf(v_hat) + ssm->epsilon);
    }
    
    // Update B matrix
    for (int i = 0; i < ssm->state_dim * ssm->input_dim; i++) {
        ssm->B_grad[i] += ssm->weight_decay * ssm->B[i];
        ssm->B_m[i] = ssm->beta1 * ssm->B_m[i] + (1.0f - ssm->beta1) * ssm->B_grad[i];
        ssm->B_v[i] = ssm->beta2 * ssm->B_v[i] + (1.0f - ssm->beta2) * ssm->B_grad[i] * ssm->B_grad[i];
        float m_hat = ssm->B_m[i] / beta1_corrected;
        float v_hat = ssm->B_v[i] / beta2_corrected;
        ssm->B[i] -= learning_rate * m_hat / (sqrtf(v_hat) + ssm->epsilon);
    }
    
    // Update C matrix
    for (int i = 0; i < ssm->output_dim * ssm->state_dim; i++) {
        ssm->C_grad[i] += ssm->weight_decay * ssm->C[i];
        ssm->C_m[i] = ssm->beta1 * ssm->C_m[i] + (1.0f - ssm->beta1) * ssm->C_grad[i];
        ssm->C_v[i] = ssm->beta2 * ssm->C_v[i] + (1.0f - ssm->beta2) * ssm->C_grad[i] * ssm->C_grad[i];
        float m_hat = ssm->C_m[i] / beta1_corrected;
        float v_hat = ssm->C_v[i] / beta2_corrected;
        ssm->C[i] -= learning_rate * m_hat / (sqrtf(v_hat) + ssm->epsilon);
    }
    
    // Update D matrix
    for (int i = 0; i < ssm->output_dim * ssm->input_dim; i++) {
        ssm->D_grad[i] += ssm->weight_decay * ssm->D[i];
        ssm->D_m[i] = ssm->beta1 * ssm->D_m[i] + (1.0f - ssm->beta1) * ssm->D_grad[i];
        ssm->D_v[i] = ssm->beta2 * ssm->D_v[i] + (1.0f - ssm->beta2) * ssm->D_grad[i] * ssm->D_grad[i];
        float m_hat = ssm->D_m[i] / beta1_corrected;
        float v_hat = ssm->D_v[i] / beta2_corrected;
        ssm->D[i] -= learning_rate * m_hat / (sqrtf(v_hat) + ssm->epsilon);
    }
}

// Save model to binary file
void save_ssm(SSM* ssm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->seq_len, sizeof(int), 1, file);
    
    // Write Adam parameters
    fwrite(&ssm->beta1, sizeof(float), 1, file);
    fwrite(&ssm->beta2, sizeof(float), 1, file);
    fwrite(&ssm->epsilon, sizeof(float), 1, file);
    fwrite(&ssm->t, sizeof(int), 1, file);
    fwrite(&ssm->weight_decay, sizeof(float), 1, file);
    
    // Write matrices
    fwrite(ssm->A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Write Adam state
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

// Load model from binary file
SSM* load_ssm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, state_dim, output_dim, seq_len;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    
    // Create SSM with custom batch size
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, custom_batch_size);
    
    // Read Adam parameters
    fread(&ssm->beta1, sizeof(float), 1, file);
    fread(&ssm->beta2, sizeof(float), 1, file);
    fread(&ssm->epsilon, sizeof(float), 1, file);
    fread(&ssm->t, sizeof(int), 1, file);
    fread(&ssm->weight_decay, sizeof(float), 1, file);
    
    // Read matrices
    fread(ssm->A, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    
    // Read Adam state
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