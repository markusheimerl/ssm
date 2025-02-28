#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // State transition matrices
    float* A;          // state_dim x state_dim (raw parameters)
    float* A_stable;   // state_dim x state_dim (stabilized version for computation)
    float* B;          // state_dim x input_dim
    float* C;          // output_dim x state_dim
    float* D;          // output_dim x input_dim

    // Gradients
    float* A_grad;
    float* B_grad;
    float* C_grad;
    float* D_grad;

    // Adam parameters
    float* A_m;
    float* A_v;
    float* B_m;
    float* B_v;
    float* C_m;
    float* C_v;
    float* D_m;
    float* D_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;

    // Helper arrays
    float* state;          // batch_size x state_dim
    float* next_state;     // batch_size x state_dim
    float* pre_state;      // batch_size x state_dim (pre-activation)
    float* predictions;    // batch_size x output_dim
    float* error;          // batch_size x output_dim
    float* state_error;    // batch_size x state_dim

    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int batch_size;
} SSM;

// Compute stable A matrix using tanh parameterization (same as GPU)
void compute_stable_A(float* A_stable, const float* A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (i == j) {
                // Diagonal elements: scaled tanh for eigenvalue control
                A_stable[idx] = 0.9f * tanhf(A[idx]);
            } else {
                // Off-diagonal elements: scaled by matrix size
                A_stable[idx] = A[idx] / sqrtf((float)n);
            }
        }
    }
}

// Compute gradient for A from gradient of A_stable (same as GPU)
void compute_A_grad_from_stable_grad(float* A_grad, const float* A_stable_grad, const float* A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (i == j) {
                // Diagonal derivative: d(tanh)/dA = sech²(A)
                float tanh_val = tanhf(A[idx]);
                float sech_squared = 1.0f - tanh_val * tanh_val;
                A_grad[idx] = A_stable_grad[idx] * 0.9f * sech_squared;
            } else {
                // Off-diagonal derivative: 1/sqrt(n)
                A_grad[idx] = A_stable_grad[idx] / sqrtf((float)n);
            }
        }
    }
}

void ssm_swish_forward(float* x, int size) {
    for (int i = 0; i < size; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        x[i] = x[i] * sigmoid;
    }
}

void ssm_swish_backward(float* grad_output, float* x, float* activated, int size) {
    for (int i = 0; i < size; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        float swish = activated[i];
        grad_output[i] *= (swish + sigmoid * (1.0f - swish));
    }
}

SSM* init_ssm(int input_dim, int state_dim, int output_dim, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    
    // Store dimensions
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->batch_size = batch_size;
    
    // Initialize Adam parameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->t = 0;
    ssm->weight_decay = 0.01f;
    
    // Allocate matrices
    ssm->A = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->A_stable = (float*)malloc(state_dim * state_dim * sizeof(float));
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
    ssm->state = (float*)calloc(batch_size * state_dim, sizeof(float));
    ssm->next_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->pre_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    ssm->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(batch_size * state_dim * sizeof(float));
    
    // Initialize matrices with scaled random values
    float scale_A = 0.1f; // Small initialization for A (similar to GPU)
    float scale_B = 1.0f / sqrtf(input_dim);
    float scale_C = 1.0f / sqrtf(state_dim);
    float scale_D = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->A[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_A;
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
    
    // Compute initial stable A
    compute_stable_A(ssm->A_stable, ssm->A, state_dim);
    
    return ssm;
}

void free_ssm(SSM* ssm) {
    free(ssm->A);
    free(ssm->A_stable);
    free(ssm->B);
    free(ssm->C);
    free(ssm->D);
    free(ssm->A_grad);
    free(ssm->B_grad);
    free(ssm->C_grad);
    free(ssm->D_grad);
    free(ssm->A_m);
    free(ssm->A_v);
    free(ssm->B_m);
    free(ssm->B_v);
    free(ssm->C_m);
    free(ssm->C_v);
    free(ssm->D_m);
    free(ssm->D_v);
    free(ssm->state);
    free(ssm->next_state);
    free(ssm->pre_state);
    free(ssm->predictions);
    free(ssm->error);
    free(ssm->state_error);
    free(ssm);
}

void ssm_forward_pass(SSM* ssm, float* X) {
    // Compute stabilized A matrix
    compute_stable_A(ssm->A_stable, ssm->A, ssm->state_dim);
    
    // Next state: x[t+1] = f(A_stable*x[t] + Bu[t]) where f is swish activation
    // State update from A_stable
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->state_dim, ssm->state_dim,
                1.0f, ssm->state, ssm->state_dim,
                ssm->A_stable, ssm->state_dim,
                0.0f, ssm->pre_state, ssm->state_dim);
    
    // Add input contribution from B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->state_dim, ssm->input_dim,
                1.0f, X, ssm->input_dim,
                ssm->B, ssm->state_dim,
                1.0f, ssm->pre_state, ssm->state_dim);
    
    // Copy pre-activation for backward pass
    memcpy(ssm->next_state, ssm->pre_state, 
           ssm->batch_size * ssm->state_dim * sizeof(float));
    
    // Apply non-linearity to state
    ssm_swish_forward(ssm->next_state, ssm->batch_size * ssm->state_dim);
    
    // Output: y[t] = Cf(x[t]) + Du[t]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->output_dim, ssm->state_dim,
                1.0f, ssm->next_state, ssm->state_dim,
                ssm->C, ssm->output_dim,
                0.0f, ssm->predictions, ssm->output_dim);
    
    // Add direct feedthrough from D
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->output_dim, ssm->input_dim,
                1.0f, X, ssm->input_dim,
                ssm->D, ssm->output_dim,
                1.0f, ssm->predictions, ssm->output_dim);
    
    // Update state
    memcpy(ssm->state, ssm->next_state, 
           ssm->batch_size * ssm->state_dim * sizeof(float));
}

float ssm_calculate_loss(SSM* ssm, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < ssm->batch_size * ssm->output_dim; i++) {
        ssm->error[i] = ssm->predictions[i] - y[i];
        loss += ssm->error[i] * ssm->error[i];
    }
    return loss / (ssm->batch_size * ssm->output_dim);
}

void ssm_zero_gradients(SSM* ssm) {
    memset(ssm->A_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float));
    memset(ssm->B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float));
    memset(ssm->C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float));
    memset(ssm->D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float));
}

void ssm_backward_pass(SSM* ssm, float* X) {
    float* A_stable_grad = (float*)malloc(ssm->state_dim * ssm->state_dim * sizeof(float));
    memset(A_stable_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float));
    
    // Gradient for C: ∂L/∂C = (∂L/∂y)(next_state)ᵀ
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->state_dim, ssm->output_dim, ssm->batch_size,
                1.0f, ssm->next_state, ssm->state_dim,
                ssm->error, ssm->output_dim,
                1.0f, ssm->C_grad, ssm->output_dim);
    
    // Gradient for D: ∂L/∂D = (∂L/∂y)Xᵀ
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->input_dim, ssm->output_dim, ssm->batch_size,
                1.0f, X, ssm->input_dim,
                ssm->error, ssm->output_dim,
                1.0f, ssm->D_grad, ssm->output_dim);
    
    // State error: ∂L/∂next_state = (∂L/∂y)Cᵀ
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim, ssm->output_dim,
                1.0f, ssm->error, ssm->output_dim,
                ssm->C, ssm->output_dim,
                0.0f, ssm->state_error, ssm->state_dim);
    
    // Apply activation gradient through swish
    ssm_swish_backward(ssm->state_error, ssm->pre_state, ssm->next_state,
                  ssm->batch_size * ssm->state_dim);
    
    // Gradient for A_stable: ∂L/∂A_stable = stateᵀ(∂L/∂pre_state)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->state_dim, ssm->state_dim, ssm->batch_size,
                1.0f, ssm->state, ssm->state_dim,
                ssm->state_error, ssm->state_dim,
                0.0f, A_stable_grad, ssm->state_dim);
    
    // Convert A_stable gradient to A gradient
    compute_A_grad_from_stable_grad(ssm->A_grad, A_stable_grad, ssm->A, ssm->state_dim);
    
    // Gradient for B: ∂L/∂B = Xᵀ(∂L/∂pre_state)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->input_dim, ssm->state_dim, ssm->batch_size,
                1.0f, X, ssm->input_dim,
                ssm->state_error, ssm->state_dim,
                1.0f, ssm->B_grad, ssm->state_dim);
    
    free(A_stable_grad);
}

void ssm_update_weights(SSM* ssm, float learning_rate) {
    ssm->t++;
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    #define UPDATE_MATRIX(W, W_grad, M, V, size) do { \
        for (int i = 0; i < size; i++) { \
            float grad = W_grad[i] / ssm->batch_size; \
            M[i] = ssm->beta1 * M[i] + (1.0f - ssm->beta1) * grad; \
            V[i] = ssm->beta2 * V[i] + (1.0f - ssm->beta2) * grad * grad; \
            float update = alpha_t * M[i] / (sqrtf(V[i]) + ssm->epsilon); \
            W[i] = W[i] * (1.0f - learning_rate * ssm->weight_decay) - update; \
        } \
    } while(0)
    
    UPDATE_MATRIX(ssm->A, ssm->A_grad, ssm->A_m, ssm->A_v, 
                 ssm->state_dim * ssm->state_dim);
    UPDATE_MATRIX(ssm->B, ssm->B_grad, ssm->B_m, ssm->B_v, 
                 ssm->state_dim * ssm->input_dim);
    UPDATE_MATRIX(ssm->C, ssm->C_grad, ssm->C_m, ssm->C_v, 
                 ssm->output_dim * ssm->state_dim);
    UPDATE_MATRIX(ssm->D, ssm->D_grad, ssm->D_m, ssm->D_v, 
                 ssm->output_dim * ssm->input_dim);
    
    #undef UPDATE_MATRIX
    
    // Update A_stable after modifying A
    compute_stable_A(ssm->A_stable, ssm->A, ssm->state_dim);
}

void save_ssm(SSM* ssm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
    fwrite(ssm->A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
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

SSM* load_ssm(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int input_dim, state_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    fread(ssm->A, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    
    fread(&ssm->t, sizeof(int), 1, file);
    fread(ssm->A_m, sizeof(float), state_dim * state_dim, file);
    fread(ssm->A_v, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B_m, sizeof(float), state_dim * input_dim, file);
    fread(ssm->B_v, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C_m, sizeof(float), output_dim * state_dim, file);
    fread(ssm->C_v, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D_m, sizeof(float), output_dim * input_dim, file);
    fread(ssm->D_v, sizeof(float), output_dim * input_dim, file);
    
    // Compute stable A matrix after loading
    compute_stable_A(ssm->A_stable, ssm->A, state_dim);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return ssm;
}

#endif