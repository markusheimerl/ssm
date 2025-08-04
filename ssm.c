#include "ssm.h"

// Apply Givens rotation to matrix at positions (i,j)
void apply_givens_rotation(float* matrix, int n, int i, int j, float cos_theta, float sin_theta) {
    for (int k = 0; k < n; k++) {
        float a_ik = matrix[i * n + k];
        float a_jk = matrix[j * n + k];
        matrix[i * n + k] = cos_theta * a_ik - sin_theta * a_jk;
        matrix[j * n + k] = sin_theta * a_ik + cos_theta * a_jk;
    }
}

// Build orthogonal matrix from rotation angles using sequential Givens rotations
void build_orthogonal_from_angles(SSM* ssm) {
    int n = ssm->state_dim;
    
    // Initialize A_orthogonal to identity matrix
    memset(ssm->A_orthogonal, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        ssm->A_orthogonal[i * n + i] = 1.0f;
    }
    
    // Apply Givens rotations
    int angle_idx = 0;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            float theta = ssm->rotation_angles[angle_idx++];
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);
            apply_givens_rotation(ssm->A_orthogonal, n, i, j, cos_theta, sin_theta);
        }
    }
}

// Compute gradients of rotation angles from gradients of orthogonal matrix
void compute_rotation_gradients(SSM* ssm, float* A_orthogonal_grad) {
    int n = ssm->state_dim;
    int num_angles = n * (n - 1) / 2;
    
    // Initialize rotation angle gradients to zero
    memset(ssm->rotation_angles_grad, 0, num_angles * sizeof(float));
    
    // Use pre-allocated temporary buffer instead of malloc
    float* A_temp = ssm->A_temp;
    memcpy(A_temp, ssm->A_orthogonal, n * n * sizeof(float));
    
    // Apply chain rule through each Givens rotation in reverse order
    int angle_idx = num_angles - 1;
    for (int i = n - 1; i >= 1; i--) {
        for (int j = i - 1; j >= 0; j--) {
            float theta = ssm->rotation_angles[angle_idx];
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);
            
            // Compute derivative of loss w.r.t. this rotation angle
            float grad_theta = 0.0f;
            for (int k = 0; k < n; k++) {
                // Derivative of G * R w.r.t. theta where G is the accumulated gradient
                // and R is the current Givens rotation
                float grad_i_k = A_orthogonal_grad[i * n + k];
                float grad_j_k = A_orthogonal_grad[j * n + k];
                float a_i_k = A_temp[i * n + k];
                float a_j_k = A_temp[j * n + k];
                
                // ∂/∂θ (cos(θ) * a_ik - sin(θ) * a_jk) = -sin(θ) * a_ik - cos(θ) * a_jk
                // ∂/∂θ (sin(θ) * a_ik + cos(θ) * a_jk) = cos(θ) * a_ik - sin(θ) * a_jk
                grad_theta += grad_i_k * (-sin_theta * a_i_k - cos_theta * a_j_k);
                grad_theta += grad_j_k * (cos_theta * a_i_k - sin_theta * a_j_k);
            }
            
            ssm->rotation_angles_grad[angle_idx] = grad_theta;
            
            // Update gradients by applying the transpose of this Givens rotation
            for (int k = 0; k < n; k++) {
                float grad_i_k = A_orthogonal_grad[i * n + k];
                float grad_j_k = A_orthogonal_grad[j * n + k];
                A_orthogonal_grad[i * n + k] = cos_theta * grad_i_k + sin_theta * grad_j_k;
                A_orthogonal_grad[j * n + k] = -sin_theta * grad_i_k + cos_theta * grad_j_k;
            }
            
            // Also apply transpose to the accumulated matrix for next iteration
            apply_givens_rotation(A_temp, n, i, j, cos_theta, sin_theta);
            
            angle_idx--;
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
    
    // Calculate number of rotation angles: n(n-1)/2
    int num_angles = state_dim * (state_dim - 1) / 2;
    
    // Allocate state space matrices
    ssm->A = (float*)calloc(state_dim * state_dim, sizeof(float)); // Keep for compatibility
    ssm->rotation_angles = (float*)malloc(num_angles * sizeof(float));
    ssm->A_orthogonal = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate gradients
    ssm->A_grad = (float*)malloc(state_dim * state_dim * sizeof(float)); // Keep for compatibility
    ssm->rotation_angles_grad = (float*)malloc(num_angles * sizeof(float));
    ssm->B_grad = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate Adam buffers
    ssm->rotation_angles_m = (float*)calloc(num_angles, sizeof(float));
    ssm->rotation_angles_v = (float*)calloc(num_angles, sizeof(float));
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
    
    // Allocate temporary buffers for gradient computation (to avoid malloc/free in backward pass)
    ssm->A_temp = (float*)malloc(state_dim * state_dim * sizeof(float));
    
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
    
    // Initialize rotation angles randomly in [-π/4, π/4] for stability
    for (int i = 0; i < num_angles; i++) {
        ssm->rotation_angles[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * (M_PI / 4.0f);
    }
    
    // Build initial orthogonal A matrix from rotation angles
    build_orthogonal_from_angles(ssm);
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    free(ssm->A); free(ssm->rotation_angles); free(ssm->A_orthogonal);
    free(ssm->B); free(ssm->C); free(ssm->D);
    free(ssm->A_grad); free(ssm->rotation_angles_grad);
    free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->rotation_angles_m); free(ssm->rotation_angles_v);
    free(ssm->B_m); free(ssm->B_v);
    free(ssm->C_m); free(ssm->C_v); free(ssm->D_m); free(ssm->D_v);
    free(ssm->states); free(ssm->predictions); free(ssm->error); free(ssm->state_error);
    free(ssm->state_outputs);
    free(ssm->A_temp);  // Free temporary buffer
    free(ssm);
}

// Reset hidden states to zero
void reset_state_ssm(SSM* ssm) {
    memset(ssm->states, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
}

// Forward pass
void forward_pass_ssm(SSM* ssm, float* X_t, int timestep) {
    // Build orthogonal A matrix from current rotation angles
    build_orthogonal_from_angles(ssm);
    
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
    int num_angles = ssm->state_dim * (ssm->state_dim - 1) / 2;
    memset(ssm->rotation_angles_grad, 0, num_angles * sizeof(float));
    memset(ssm->B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float));
    memset(ssm->C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float));
    memset(ssm->D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float));
}

// Backward pass
void backward_pass_ssm(SSM* ssm, float* X) {
    // Clear state errors
    memset(ssm->state_error, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    
    // Allocate temporary gradient for A_orthogonal
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
    
    // Compute gradients w.r.t. rotation angles using chain rule
    compute_rotation_gradients(ssm, A_orthogonal_grad);
    
    free(A_orthogonal_grad);
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int num_angles = ssm->state_dim * (ssm->state_dim - 1) / 2;
    
    // Update rotation_angles
    for (int i = 0; i < num_angles; i++) {
        float grad = ssm->rotation_angles_grad[i] / ssm->batch_size;
        ssm->rotation_angles_m[i] = ssm->beta1 * ssm->rotation_angles_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->rotation_angles_v[i] = ssm->beta2 * ssm->rotation_angles_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->rotation_angles_m[i] / (sqrtf(ssm->rotation_angles_v[i]) + ssm->epsilon);
        ssm->rotation_angles[i] = ssm->rotation_angles[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
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
    
    int num_angles = ssm->state_dim * (ssm->state_dim - 1) / 2;
    
    // Save rotation angles and matrices
    fwrite(ssm->rotation_angles, sizeof(float), num_angles, file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    // Save Adam state
    fwrite(&ssm->t, sizeof(int), 1, file);
    fwrite(ssm->rotation_angles_m, sizeof(float), num_angles, file);
    fwrite(ssm->rotation_angles_v, sizeof(float), num_angles, file);
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
    int num_angles = state_dim * (state_dim - 1) / 2;
    
    // Initialize model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Load rotation angles and matrices
    fread(ssm->rotation_angles, sizeof(float), num_angles, file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    
    // Load Adam state
    fread(&ssm->t, sizeof(int), 1, file);
    fread(ssm->rotation_angles_m, sizeof(float), num_angles, file);
    fread(ssm->rotation_angles_v, sizeof(float), num_angles, file);
    fread(ssm->B_m, sizeof(float), state_dim * input_dim, file);
    fread(ssm->B_v, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C_m, sizeof(float), output_dim * state_dim, file);
    fread(ssm->C_v, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D_m, sizeof(float), output_dim * input_dim, file);
    fread(ssm->D_v, sizeof(float), output_dim * input_dim, file);
    
    // Build orthogonal A matrix from loaded rotation angles
    build_orthogonal_from_angles(ssm);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}
