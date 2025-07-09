#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // State space matrices
    float* A;           // state_dim x state_dim (state transition)
    float* B;           // state_dim x input_dim (input to state)
    float* C;           // output_dim x state_dim (state to output)
    float* D;           // output_dim x input_dim (input to output)
    
    // Gradients
    float* A_grad;      // state_dim x state_dim
    float* B_grad;      // state_dim x input_dim
    float* C_grad;      // output_dim x state_dim
    float* D_grad;      // output_dim x input_dim
    
    // Adam parameters for A, B, C, D
    float* A_m; float* A_v;
    float* B_m; float* B_v;
    float* C_m; float* C_v;
    float* D_m; float* D_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Helper arrays for forward/backward pass
    float* states;          // batch_size x seq_len x state_dim
    float* predictions;     // batch_size x seq_len x output_dim
    float* error;          // batch_size x seq_len x output_dim
    float* state_error;    // batch_size x seq_len x state_dim
    float* pre_activation_states;  // batch_size x seq_len x state_dim (for Swish)
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} SSM;

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
    ssm->A = (float*)malloc(state_dim * state_dim * sizeof(float));
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
    
    // Allocate helper arrays
    ssm->states = (float*)malloc(batch_size * seq_len * state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(batch_size * seq_len * output_dim * sizeof(float));
    ssm->error = (float*)malloc(batch_size * seq_len * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(batch_size * seq_len * state_dim * sizeof(float));
    ssm->pre_activation_states = (float*)malloc(batch_size * seq_len * state_dim * sizeof(float));
    
    // Initialize matrices with careful scaling for stability but larger outputs
    float scale_B = 0.5f / sqrt(input_dim);
    float scale_C = 0.5f / sqrt(state_dim);
    float scale_D = 0.1f / sqrt(input_dim);
    
    // Initialize A as a stable matrix with eigenvalues < 1
    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->A[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * 0.05f;
    }
    
    // Add diagonal stability
    for (int i = 0; i < state_dim; i++) {
        ssm->A[i * state_dim + i] = 0.5f + ((float)rand() / (float)RAND_MAX * 0.3f);
    }
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->B[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->C[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->D[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_D;
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
    free(ssm->pre_activation_states);
    free(ssm);
}

// Forward pass
void forward_pass_ssm(SSM* ssm, float* X) {
    // Clear states
    memset(ssm->states, 0, ssm->batch_size * ssm->seq_len * ssm->state_dim * sizeof(float));
    
    for (int b = 0; b < ssm->batch_size; b++) {
        for (int t = 0; t < ssm->seq_len; t++) {
            int x_idx = b * ssm->seq_len * ssm->input_dim + t * ssm->input_dim;
            int h_idx = b * ssm->seq_len * ssm->state_dim + t * ssm->state_dim;
            int y_idx = b * ssm->seq_len * ssm->output_dim + t * ssm->output_dim;
            
            // h_t = A * h_{t-1} + B * x_t
            if (t > 0) {
                int h_prev_idx = b * ssm->seq_len * ssm->state_dim + (t-1) * ssm->state_dim;
                // h_t = A * h_{t-1}
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                           ssm->state_dim, ssm->state_dim, 1.0f,
                           ssm->A, ssm->state_dim,
                           &ssm->states[h_prev_idx], 1,
                           0.0f, &ssm->states[h_idx], 1);
            }
            
            // h_t += B * x_t
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                       ssm->state_dim, ssm->input_dim, 1.0f,
                       ssm->B, ssm->input_dim,
                       &X[x_idx], 1,
                       1.0f, &ssm->states[h_idx], 1);
            
            // Store pre-activation for backward pass
            memcpy(&ssm->pre_activation_states[h_idx], &ssm->states[h_idx], 
                   ssm->state_dim * sizeof(float));
            
            // Apply Swish: h_t = h_t * σ(h_t)
            for (int i = 0; i < ssm->state_dim; i++) {
                float x = ssm->states[h_idx + i];
                float sigmoid = 1.0f / (1.0f + expf(-x));
                ssm->states[h_idx + i] = x * sigmoid;
            }
            
            // y_t = C * h_t + D * x_t
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                       ssm->output_dim, ssm->state_dim, 1.0f,
                       ssm->C, ssm->state_dim,
                       &ssm->states[h_idx], 1,
                       0.0f, &ssm->predictions[y_idx], 1);
            
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                       ssm->output_dim, ssm->input_dim, 1.0f,
                       ssm->D, ssm->input_dim,
                       &X[x_idx], 1,
                       1.0f, &ssm->predictions[y_idx], 1);
        }
    }
}

// Calculate loss
float calculate_loss_ssm(SSM* ssm, float* y) {
    float loss = 0.0f;
    int total_size = ssm->batch_size * ssm->seq_len * ssm->output_dim;
    
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

// Backward pass with Swish derivative
void backward_pass_ssm(SSM* ssm, float* X) {
    // Clear state errors
    memset(ssm->state_error, 0, ssm->batch_size * ssm->seq_len * ssm->state_dim * sizeof(float));
    
    for (int b = 0; b < ssm->batch_size; b++) {
        // Backward pass through time
        for (int t = ssm->seq_len - 1; t >= 0; t--) {
            int x_idx = b * ssm->seq_len * ssm->input_dim + t * ssm->input_dim;
            int h_idx = b * ssm->seq_len * ssm->state_dim + t * ssm->state_dim;
            int y_idx = b * ssm->seq_len * ssm->output_dim + t * ssm->output_dim;
            
            // ∂L/∂C += ∂L/∂y_t * h_t^T
            cblas_sger(CblasRowMajor,
                      ssm->output_dim, ssm->state_dim, 1.0f,
                      &ssm->error[y_idx], 1,
                      &ssm->states[h_idx], 1,
                      ssm->C_grad, ssm->state_dim);
            
            // ∂L/∂D += ∂L/∂y_t * x_t^T
            cblas_sger(CblasRowMajor,
                      ssm->output_dim, ssm->input_dim, 1.0f,
                      &ssm->error[y_idx], 1,
                      &X[x_idx], 1,
                      ssm->D_grad, ssm->input_dim);
            
            // ∂L/∂h_t = C^T * ∂L/∂y_t
            cblas_sgemv(CblasRowMajor, CblasTrans,
                       ssm->output_dim, ssm->state_dim, 1.0f,
                       ssm->C, ssm->state_dim,
                       &ssm->error[y_idx], 1,
                       0.0f, &ssm->state_error[h_idx], 1);
            
            // Add error from future time step if not last
            if (t < ssm->seq_len - 1) {
                int h_next_idx = b * ssm->seq_len * ssm->state_dim + (t+1) * ssm->state_dim;
                cblas_sgemv(CblasRowMajor, CblasTrans,
                           ssm->state_dim, ssm->state_dim, 1.0f,
                           ssm->A, ssm->state_dim,
                           &ssm->state_error[h_next_idx], 1,
                           1.0f, &ssm->state_error[h_idx], 1);
            }
            
            // Apply Swish derivative: ∂L/∂h_pre = ∂L/∂h_post * [σ(h_pre) + h_pre * σ(h_pre) * (1 - σ(h_pre))]
            for (int i = 0; i < ssm->state_dim; i++) {
                float x = ssm->pre_activation_states[h_idx + i];
                float sigmoid = 1.0f / (1.0f + expf(-x));
                float swish_derivative = sigmoid + x * sigmoid * (1.0f - sigmoid);
                ssm->state_error[h_idx + i] *= swish_derivative;
            }
            
            // ∂L/∂B += ∂L/∂h_t * x_t^T
            cblas_sger(CblasRowMajor,
                      ssm->state_dim, ssm->input_dim, 1.0f,
                      &ssm->state_error[h_idx], 1,
                      &X[x_idx], 1,
                      ssm->B_grad, ssm->input_dim);
            
            // ∂L/∂A += ∂L/∂h_t * h_{t-1}^T
            if (t > 0) {
                int h_prev_idx = b * ssm->seq_len * ssm->state_dim + (t-1) * ssm->state_dim;
                cblas_sger(CblasRowMajor,
                          ssm->state_dim, ssm->state_dim, 1.0f,
                          &ssm->state_error[h_idx], 1,
                          &ssm->states[h_prev_idx], 1,
                          ssm->A_grad, ssm->state_dim);
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

#endif