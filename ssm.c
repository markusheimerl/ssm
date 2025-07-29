#include "ssm.h"

// Initialize the state space model
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size, int mlp_hidden_dim) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    
    // Store dimensions
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->seq_len = seq_len;
    ssm->batch_size = batch_size;
    ssm->mlp_hidden_dim = mlp_hidden_dim;
    
    // Initialize Adam parameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->t = 0;
    ssm->weight_decay = 0.001f;
    
    // Allocate state space matrices
    ssm->A = (float*)calloc(state_dim * state_dim, sizeof(float));  // unused but kept
    ssm->B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate MLP matrices
    ssm->W1 = (float*)malloc(mlp_hidden_dim * input_dim * sizeof(float));
    ssm->W2 = (float*)malloc(state_dim * state_dim * mlp_hidden_dim * sizeof(float));
    
    // Allocate gradients
    ssm->A_grad = (float*)malloc(state_dim * state_dim * sizeof(float));  // unused but kept
    ssm->B_grad = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    ssm->W1_grad = (float*)malloc(mlp_hidden_dim * input_dim * sizeof(float));
    ssm->W2_grad = (float*)malloc(state_dim * state_dim * mlp_hidden_dim * sizeof(float));
    
    // Allocate Adam buffers
    ssm->A_m = (float*)calloc(state_dim * state_dim, sizeof(float));  // unused but kept
    ssm->A_v = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->B_m = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->B_v = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->C_m = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->C_v = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->D_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->D_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->W1_m = (float*)calloc(mlp_hidden_dim * input_dim, sizeof(float));
    ssm->W1_v = (float*)calloc(mlp_hidden_dim * input_dim, sizeof(float));
    ssm->W2_m = (float*)calloc(state_dim * state_dim * mlp_hidden_dim, sizeof(float));
    ssm->W2_v = (float*)calloc(state_dim * state_dim * mlp_hidden_dim, sizeof(float));
    
    // Allocate helper arrays (time-major format)
    ssm->states = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->error = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->state_outputs = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    
    // Allocate MLP helper arrays
    ssm->Z_t = (float*)malloc(batch_size * mlp_hidden_dim * sizeof(float));
    ssm->U_t = (float*)malloc(batch_size * mlp_hidden_dim * sizeof(float));
    ssm->A_t = (float*)malloc(batch_size * state_dim * state_dim * sizeof(float));
    ssm->Z_error = (float*)malloc(batch_size * mlp_hidden_dim * sizeof(float));
    ssm->U_error = (float*)malloc(batch_size * mlp_hidden_dim * sizeof(float));
    
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
    
    // Initialize MLP matrices
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 0.1f / sqrtf(mlp_hidden_dim);  // smaller for tanh stability
    
    for (int i = 0; i < mlp_hidden_dim * input_dim; i++) {
        ssm->W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < state_dim * state_dim * mlp_hidden_dim; i++) {
        ssm->W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    return ssm;
}

// Free memory
void free_ssm(SSM* ssm) {
    free(ssm->A); free(ssm->B); free(ssm->C); free(ssm->D);
    free(ssm->W1); free(ssm->W2);
    free(ssm->A_grad); free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->W1_grad); free(ssm->W2_grad);
    free(ssm->A_m); free(ssm->A_v); free(ssm->B_m); free(ssm->B_v);
    free(ssm->C_m); free(ssm->C_v); free(ssm->D_m); free(ssm->D_v);
    free(ssm->W1_m); free(ssm->W1_v); free(ssm->W2_m); free(ssm->W2_v);
    free(ssm->states); free(ssm->predictions); free(ssm->error); free(ssm->state_error);
    free(ssm->state_outputs);
    free(ssm->Z_t); free(ssm->U_t); free(ssm->A_t); free(ssm->Z_error); free(ssm->U_error);
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
    
    // Step 1: Z_t = X_t W_1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->mlp_hidden_dim, ssm->input_dim,
                1.0f, X_t, ssm->input_dim,
                ssm->W1, ssm->input_dim,
                0.0f, ssm->Z_t, ssm->mlp_hidden_dim);
    
    // Step 2: U_t = σ(Z_t ⊙ σ(Z_t))
    for (int i = 0; i < ssm->batch_size * ssm->mlp_hidden_dim; i++) {
        float z = ssm->Z_t[i];
        float sigmoid_z = 1.0f / (1.0f + expf(-z));
        ssm->U_t[i] = sigmoid_z * (z * sigmoid_z);
    }
    
    // Step 3: A_t = tanh(U_t W_2) - reshape U_t W_2 into batch_size x state_dim x state_dim
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim * ssm->state_dim, ssm->mlp_hidden_dim,
                1.0f, ssm->U_t, ssm->mlp_hidden_dim,
                ssm->W2, ssm->mlp_hidden_dim,
                0.0f, ssm->A_t, ssm->state_dim * ssm->state_dim);
    
    // Apply tanh to get final A_t
    for (int i = 0; i < ssm->batch_size * ssm->state_dim * ssm->state_dim; i++) {
        ssm->A_t[i] = tanhf(ssm->A_t[i]);
    }
        
    // Step 4: H_t = X_t B^T + H_{t-1} A_t^T
    // First compute X_t B^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim, ssm->input_dim,
                1.0f, X_t, ssm->input_dim,
                ssm->B, ssm->input_dim,
                0.0f, h_t, ssm->state_dim);
    
    // Then add H_{t-1} A_t^T (batch-wise matrix multiplication)
    if (timestep > 0) {
        for (int b = 0; b < ssm->batch_size; b++) {
            float* h_prev_b = h_prev + b * ssm->state_dim;
            float* A_t_b = ssm->A_t + b * ssm->state_dim * ssm->state_dim;
            float* h_t_b = h_t + b * ssm->state_dim;
            
            // h_t_b += h_prev_b * A_t_b^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, ssm->state_dim, ssm->state_dim,
                        1.0f, h_prev_b, ssm->state_dim,
                        A_t_b, ssm->state_dim,
                        1.0f, h_t_b, ssm->state_dim);
        }
    }
    
    // Step 5: O_t = H_t ⊙ σ(H_t) 
    for (int i = 0; i < ssm->batch_size * ssm->state_dim; i++) {
        float h = h_t[i];
        float sigmoid_h = 1.0f / (1.0f + expf(-h));
        o_t[i] = h * sigmoid_h;
    }
    
    // Step 6: Y_t = O_t C^T + X_t D^T
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
    memset(ssm->W1_grad, 0, ssm->mlp_hidden_dim * ssm->input_dim * sizeof(float));
    memset(ssm->W2_grad, 0, ssm->state_dim * ssm->state_dim * ssm->mlp_hidden_dim * sizeof(float));
}

// Backward pass
void backward_pass_ssm(SSM* ssm, float* X) {
    // Clear state errors and MLP errors
    memset(ssm->state_error, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    memset(ssm->Z_error, 0, ssm->batch_size * ssm->mlp_hidden_dim * sizeof(float));
    memset(ssm->U_error, 0, ssm->batch_size * ssm->mlp_hidden_dim * sizeof(float));
    
    for (int t = ssm->seq_len - 1; t >= 0; t--) {
        float* X_t = X + t * ssm->batch_size * ssm->input_dim;
        float* h_t = ssm->states + t * ssm->batch_size * ssm->state_dim;
        float* o_t = ssm->state_outputs + t * ssm->batch_size * ssm->state_dim;
        float* dy_t = ssm->error + t * ssm->batch_size * ssm->output_dim;
        float* dh_t = ssm->state_error + t * ssm->batch_size * ssm->state_dim;
        
        // Recompute forward pass values for this timestep (needed for gradients)
        // Z_t = X_t W_1
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->mlp_hidden_dim, ssm->input_dim,
                    1.0f, X_t, ssm->input_dim,
                    ssm->W1, ssm->input_dim,
                    0.0f, ssm->Z_t, ssm->mlp_hidden_dim);
        
        // U_t = σ(Z_t ⊙ σ(Z_t))
        for (int i = 0; i < ssm->batch_size * ssm->mlp_hidden_dim; i++) {
            float z = ssm->Z_t[i];
            float sigmoid_z = 1.0f / (1.0f + expf(-z));
            ssm->U_t[i] = sigmoid_z * (z * sigmoid_z);
        }
        
        // A_t = tanh(U_t W_2)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->state_dim * ssm->state_dim, ssm->mlp_hidden_dim,
                    1.0f, ssm->U_t, ssm->mlp_hidden_dim,
                    ssm->W2, ssm->mlp_hidden_dim,
                    0.0f, ssm->A_t, ssm->state_dim * ssm->state_dim);
        
        for (int i = 0; i < ssm->batch_size * ssm->state_dim * ssm->state_dim; i++) {
            ssm->A_t[i] = tanhf(ssm->A_t[i]);
        }
        
        // Standard gradients for C, D (unchanged)
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
        
        // ∂L/∂H_t = ∂L/∂O_t ⊙ [σ(H_t) + H_t σ'(H_t)] (swish derivative)
        for (int i = 0; i < ssm->batch_size * ssm->state_dim; i++) {
            float h = h_t[i];
            float sigmoid = 1.0f / (1.0f + expf(-h));
            dh_t[i] = do_t[i] * (sigmoid + h * sigmoid * (1.0f - sigmoid));
        }
        
        // Backward through H_t = X_t B^T + H_{t-1} A_t^T
        // ∂L/∂B += (∂L/∂H_t)^T X_t
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->state_dim, ssm->input_dim, ssm->batch_size,
                    1.0f, dh_t, ssm->state_dim,
                    X_t, ssm->input_dim,
                    1.0f, ssm->B_grad, ssm->input_dim);
        
        // Compute ∂L/∂A_t and accumulate ∂L/∂H_{t-1} (batch-wise)
        if (t > 0) {
            float* h_prev = ssm->states + (t-1) * ssm->batch_size * ssm->state_dim;
            float* dh_prev = ssm->state_error + (t-1) * ssm->batch_size * ssm->state_dim;
            
            // Temporary storage for ∂L/∂A_t
            float* dA_t = (float*)malloc(ssm->batch_size * ssm->state_dim * ssm->state_dim * sizeof(float));
            
            for (int b = 0; b < ssm->batch_size; b++) {
                float* h_prev_b = h_prev + b * ssm->state_dim;
                float* dh_t_b = dh_t + b * ssm->state_dim;
                float* A_t_b = ssm->A_t + b * ssm->state_dim * ssm->state_dim;
                float* dA_t_b = dA_t + b * ssm->state_dim * ssm->state_dim;
                float* dh_prev_b = dh_prev + b * ssm->state_dim;
                
                // ∂L/∂A_t_b = (∂L/∂H_t_b)^T H_{t-1}_b
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            ssm->state_dim, ssm->state_dim, 1,
                            1.0f, dh_t_b, ssm->state_dim,
                            h_prev_b, ssm->state_dim,
                            0.0f, dA_t_b, ssm->state_dim);
                
                // ∂L/∂H_{t-1}_b += (∂L/∂H_t_b) A_t_b
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            1, ssm->state_dim, ssm->state_dim,
                            1.0f, dh_t_b, ssm->state_dim,
                            A_t_b, ssm->state_dim,
                            1.0f, dh_prev_b, ssm->state_dim);
            }
            
            // Backward through A_t = tanh(U_t W_2)
            // ∂L/∂(U_t W_2) = ∂L/∂A_t ⊙ sech²(U_t W_2) = ∂L/∂A_t ⊙ (1 - A_t²)
            float* dUW2 = (float*)malloc(ssm->batch_size * ssm->state_dim * ssm->state_dim * sizeof(float));
            for (int i = 0; i < ssm->batch_size * ssm->state_dim * ssm->state_dim; i++) {
                dUW2[i] = dA_t[i] * (1.0f - ssm->A_t[i] * ssm->A_t[i]);
            }
            
            // ∂L/∂W_2 += U_t^T (∂L/∂(U_t W_2))
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        ssm->mlp_hidden_dim, ssm->state_dim * ssm->state_dim, ssm->batch_size,
                        1.0f, ssm->U_t, ssm->mlp_hidden_dim,
                        dUW2, ssm->state_dim * ssm->state_dim,
                        1.0f, ssm->W2_grad, ssm->state_dim * ssm->state_dim);
            
            // ∂L/∂U_t = (∂L/∂(U_t W_2)) W_2^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        ssm->batch_size, ssm->mlp_hidden_dim, ssm->state_dim * ssm->state_dim,
                        1.0f, dUW2, ssm->state_dim * ssm->state_dim,
                        ssm->W2, ssm->mlp_hidden_dim,
                        0.0f, ssm->U_error, ssm->mlp_hidden_dim);
            
            free(dUW2);
            free(dA_t);
        }
        
        // Backward through U_t = σ(Z_t ⊙ σ(Z_t)) = σ(Z_t)² Z_t
        // dU/dZ = σ(Z)² + 2σ(Z)Z σ'(Z) = σ(Z)² + 2σ(Z)Z σ(Z)(1-σ(Z)) = σ(Z)[σ(Z) + 2Z(1-σ(Z))]
        for (int i = 0; i < ssm->batch_size * ssm->mlp_hidden_dim; i++) {
            float z = ssm->Z_t[i];
            float sigmoid_z = 1.0f / (1.0f + expf(-z));
            ssm->Z_error[i] = ssm->U_error[i] * sigmoid_z * (sigmoid_z + 2.0f * z * (1.0f - sigmoid_z));
        }
        
        // Backward through Z_t = X_t W_1
        // ∂L/∂W_1 += X_t^T (∂L/∂Z_t)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->mlp_hidden_dim, ssm->input_dim, ssm->batch_size,
                    1.0f, ssm->Z_error, ssm->mlp_hidden_dim,
                    X_t, ssm->input_dim,
                    1.0f, ssm->W1_grad, ssm->input_dim);
    }
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
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
    
    // Update W1
    for (int i = 0; i < ssm->mlp_hidden_dim * ssm->input_dim; i++) {
        float grad = ssm->W1_grad[i] / ssm->batch_size;
        ssm->W1_m[i] = ssm->beta1 * ssm->W1_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->W1_v[i] = ssm->beta2 * ssm->W1_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->W1_m[i] / (sqrtf(ssm->W1_v[i]) + ssm->epsilon);
        ssm->W1[i] = ssm->W1[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update W2
    for (int i = 0; i < ssm->state_dim * ssm->state_dim * ssm->mlp_hidden_dim; i++) {
        float grad = ssm->W2_grad[i] / ssm->batch_size;
        ssm->W2_m[i] = ssm->beta1 * ssm->W2_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->W2_v[i] = ssm->beta2 * ssm->W2_v[i] + (1.0f - ssm->beta2) * grad * grad;
        float update = alpha_t * ssm->W2_m[i] / (sqrtf(ssm->W2_v[i]) + ssm->epsilon);
        ssm->W2[i] = ssm->W2[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
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
    fwrite(&ssm->mlp_hidden_dim, sizeof(int), 1, file);
    
    // Save matrices
    fwrite(ssm->A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->W1, sizeof(float), ssm->mlp_hidden_dim * ssm->input_dim, file);
    fwrite(ssm->W2, sizeof(float), ssm->state_dim * ssm->state_dim * ssm->mlp_hidden_dim, file);
    
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
    fwrite(ssm->W1_m, sizeof(float), ssm->mlp_hidden_dim * ssm->input_dim, file);
    fwrite(ssm->W1_v, sizeof(float), ssm->mlp_hidden_dim * ssm->input_dim, file);
    fwrite(ssm->W2_m, sizeof(float), ssm->state_dim * ssm->state_dim * ssm->mlp_hidden_dim, file);
    fwrite(ssm->W2_v, sizeof(float), ssm->state_dim * ssm->state_dim * ssm->mlp_hidden_dim, file);
    
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
    int input_dim, state_dim, output_dim, seq_len, stored_batch_size, mlp_hidden_dim;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&mlp_hidden_dim, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize model
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size, mlp_hidden_dim);
    
    // Load matrices
    fread(ssm->A, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    fread(ssm->W1, sizeof(float), mlp_hidden_dim * input_dim, file);
    fread(ssm->W2, sizeof(float), state_dim * state_dim * mlp_hidden_dim, file);
    
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
    fread(ssm->W1_m, sizeof(float), mlp_hidden_dim * input_dim, file);
    fread(ssm->W1_v, sizeof(float), mlp_hidden_dim * input_dim, file);
    fread(ssm->W2_m, sizeof(float), state_dim * state_dim * mlp_hidden_dim, file);
    fread(ssm->W2_v, sizeof(float), state_dim * state_dim * mlp_hidden_dim, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}
