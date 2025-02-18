#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>

typedef struct {
    // State transition matrices
    float* A;  // state_dim x state_dim
    float* B;  // state_dim x input_dim
    float* C;  // output_dim x state_dim
    float* D;  // output_dim x input_dim

    // Gradients
    float* A_grad;
    float* B_grad;
    float* C_grad;
    float* D_grad;

    // Block Fisher Information Matrices
    float* A_fim;  // state_dim x state_dim
    float* B_fim;  // input_dim x input_dim
    float* C_fim;  // output_dim x output_dim
    float* D_fim;  // output_dim x input_dim

    // Pre-allocated workspaces
    float* fim_workspace;    // max_dim x max_dim
    float* eigenvals;        // max_dim
    float* eigenvecs;        // max_dim x max_dim
    float* new_fim;          // max_dim x max_dim
    float* fim_copy;         // max_dim x max_dim
    float* nat_grad_workspace; // max_dim

    // Helper arrays
    float* state;          // batch_size x state_dim
    float* next_state;     // batch_size x state_dim
    float* pre_state;      // batch_size x state_dim
    float* predictions;    // batch_size x output_dim
    float* error;          // batch_size x output_dim
    float* state_error;    // batch_size x state_dim

    // FIM parameters
    float damping;
    float weight_decay;
    float fim_ema_rate;

    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int batch_size;
    int max_dim;
} SSM;

void swish_forward(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void swish_backward(float* grad_output, float* x, int size) {
    for (int i = 0; i < size; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        float swish = x[i] * sigmoid;
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
    
    // Find maximum dimension for workspace allocation
    ssm->max_dim = input_dim;
    if (state_dim > ssm->max_dim) ssm->max_dim = state_dim;
    if (output_dim > ssm->max_dim) ssm->max_dim = output_dim;
    
    // Initialize parameters
    ssm->damping = 1e-4f;
    ssm->weight_decay = 0.01f;
    ssm->fim_ema_rate = 0.05f;
    
    // Allocate matrices
    ssm->A = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate gradients
    ssm->A_grad = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B_grad = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate block FIMs
    ssm->A_fim = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->B_fim = (float*)calloc(input_dim * input_dim, sizeof(float));
    ssm->C_fim = (float*)calloc(output_dim * output_dim, sizeof(float));
    ssm->D_fim = (float*)calloc(output_dim * input_dim, sizeof(float));
    
    // Allocate workspaces
    int max_workspace = ssm->max_dim * ssm->max_dim;
    ssm->fim_workspace = (float*)malloc(max_workspace * sizeof(float));
    ssm->eigenvals = (float*)malloc(ssm->max_dim * sizeof(float));
    ssm->eigenvecs = (float*)malloc(max_workspace * sizeof(float));
    ssm->new_fim = (float*)malloc(max_workspace * sizeof(float));
    ssm->fim_copy = (float*)malloc(max_workspace * sizeof(float));
    ssm->nat_grad_workspace = (float*)malloc(ssm->max_dim * sizeof(float));
    
    // Allocate helper arrays
    ssm->state = (float*)calloc(batch_size * state_dim, sizeof(float));
    ssm->next_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->pre_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    ssm->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(batch_size * state_dim * sizeof(float));
    
    // Initialize matrices with scaled random values
    float scale_A = 1.0f / sqrtf(state_dim);
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
    
    return ssm;
}

void free_ssm(SSM* ssm) {
    free(ssm->A);
    free(ssm->B);
    free(ssm->C);
    free(ssm->D);
    free(ssm->A_grad);
    free(ssm->B_grad);
    free(ssm->C_grad);
    free(ssm->D_grad);
    free(ssm->A_fim);
    free(ssm->B_fim);
    free(ssm->C_fim);
    free(ssm->D_fim);
    free(ssm->fim_workspace);
    free(ssm->eigenvals);
    free(ssm->eigenvecs);
    free(ssm->new_fim);
    free(ssm->fim_copy);
    free(ssm->nat_grad_workspace);
    free(ssm->state);
    free(ssm->next_state);
    free(ssm->pre_state);
    free(ssm->predictions);
    free(ssm->error);
    free(ssm->state_error);
    free(ssm);
}

void forward_pass(SSM* ssm, float* X) {
    // Next state: x[t+1] = f(Ax[t] + Bu[t]) where f is swish activation
    // State update from A
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->state_dim, ssm->state_dim,
                1.0f, ssm->state, ssm->state_dim,
                ssm->A, ssm->state_dim,
                0.0f, ssm->pre_state, ssm->state_dim);
    
    // Add input contribution from B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->state_dim, ssm->input_dim,
                1.0f, X, ssm->input_dim,
                ssm->B, ssm->state_dim,
                1.0f, ssm->pre_state, ssm->state_dim);
    
    // Apply non-linearity to state
    memcpy(ssm->next_state, ssm->pre_state, 
           ssm->batch_size * ssm->state_dim * sizeof(float));
    swish_forward(ssm->next_state, ssm->batch_size * ssm->state_dim);
    
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

float calculate_loss(SSM* ssm, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < ssm->batch_size * ssm->output_dim; i++) {
        ssm->error[i] = ssm->predictions[i] - y[i];
        loss += ssm->error[i] * ssm->error[i];
    }
    return loss / (ssm->batch_size * ssm->output_dim);
}

void zero_gradients(SSM* ssm) {
    memset(ssm->A_grad, 0, ssm->state_dim * ssm->state_dim * sizeof(float));
    memset(ssm->B_grad, 0, ssm->state_dim * ssm->input_dim * sizeof(float));
    memset(ssm->C_grad, 0, ssm->output_dim * ssm->state_dim * sizeof(float));
    memset(ssm->D_grad, 0, ssm->output_dim * ssm->input_dim * sizeof(float));
}

void backward_pass(SSM* ssm, float* X) {
    // Gradient for C: ∂L/∂C = (∂L/∂y)f(x)ᵀ
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->state_dim, ssm->output_dim, ssm->batch_size,
                1.0f, ssm->state, ssm->state_dim,
                ssm->error, ssm->output_dim,
                1.0f, ssm->C_grad, ssm->output_dim);
    
    // Gradient for D: ∂L/∂D = (∂L/∂y)uᵀ
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->input_dim, ssm->output_dim, ssm->batch_size,
                1.0f, X, ssm->input_dim,
                ssm->error, ssm->output_dim,
                1.0f, ssm->D_grad, ssm->output_dim);
    
    // State error: ∂L/∂f(x) = (∂L/∂y)Cᵀ
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim, ssm->output_dim,
                1.0f, ssm->error, ssm->output_dim,
                ssm->C, ssm->output_dim,
                0.0f, ssm->state_error, ssm->state_dim);
    
    // Apply activation gradient
    swish_backward(ssm->state_error, ssm->pre_state, 
                  ssm->batch_size * ssm->state_dim);
    
    // Gradient for A: ∂L/∂A = xᵀ(∂L/∂z)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->state_dim, ssm->state_dim, ssm->batch_size,
                1.0f, ssm->state, ssm->state_dim,
                ssm->state_error, ssm->state_dim,
                1.0f, ssm->A_grad, ssm->state_dim);
    
    // Gradient for B: ∂L/∂B = uᵀ(∂L/∂z)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->input_dim, ssm->state_dim, ssm->batch_size,
                1.0f, X, ssm->input_dim,
                ssm->state_error, ssm->state_dim,
                1.0f, ssm->B_grad, ssm->state_dim);
}

void update_block_fim(SSM* ssm, float* fim, float* grad, int rows,
                     int batch_size, float ema_rate) {
    memset(ssm->new_fim, 0, rows * rows * sizeof(float));
    
    // Compute new FIM estimate using outer product of gradients
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rows, rows, batch_size,
                1.0f/batch_size, grad, rows,
                grad, rows,
                0.0f, ssm->new_fim, rows);
    
    // Update FIM using exponential moving average
    for(int i = 0; i < rows * rows; i++) {
        fim[i] = (1.0f - ema_rate) * fim[i] + ema_rate * ssm->new_fim[i];
    }
}

void block_fim_solve(SSM* ssm, float* fim, float* grad, int rows) {
    // Copy FIM for eigendecomposition
    memcpy(ssm->fim_copy, fim, rows * rows * sizeof(float));
    
    // Compute eigendecomposition
    LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', rows, ssm->fim_copy, rows, ssm->eigenvals);
    
    // Store eigenvectors
    memcpy(ssm->eigenvecs, ssm->fim_copy, rows * rows * sizeof(float));
    
    // Step 1: V^T * grad
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rows, 1, rows,
                1.0f, ssm->eigenvecs, rows,
                grad, 1,
                0.0f, ssm->nat_grad_workspace, 1);
    
    // Step 2: Apply scaled eigenvalues
    for(int i = 0; i < rows; i++) {
        float lambda = ssm->eigenvals[i] + ssm->damping;
        if (lambda > 1e-6f) {
            ssm->nat_grad_workspace[i] /= lambda;
        } else {
            ssm->nat_grad_workspace[i] = 0.0f;
        }
    }
    
    // Step 3: V * ssm->nat_grad_workspace
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, 1, rows,
                1.0f, ssm->eigenvecs, rows,
                ssm->nat_grad_workspace, 1,
                0.0f, grad, 1);
}

void update_weights(SSM* ssm, float learning_rate) {
    // Update block FIMs
    update_block_fim(ssm, ssm->A_fim, ssm->A_grad, ssm->state_dim,
                    ssm->batch_size, ssm->fim_ema_rate);
    update_block_fim(ssm, ssm->B_fim, ssm->B_grad, ssm->input_dim,
                    ssm->batch_size, ssm->fim_ema_rate);
    update_block_fim(ssm, ssm->C_fim, ssm->C_grad, ssm->output_dim,
                    ssm->batch_size, ssm->fim_ema_rate);
    update_block_fim(ssm, ssm->D_fim, ssm->D_grad, ssm->output_dim,
                    ssm->batch_size, ssm->fim_ema_rate);
    
    // Compute natural gradients
    block_fim_solve(ssm, ssm->A_fim, ssm->A_grad, ssm->state_dim);
    block_fim_solve(ssm, ssm->B_fim, ssm->B_grad, ssm->input_dim);
    block_fim_solve(ssm, ssm->C_fim, ssm->C_grad, ssm->output_dim);
    block_fim_solve(ssm, ssm->D_fim, ssm->D_grad, ssm->output_dim);
    
    // Apply updates with weight decay
    #define UPDATE_MATRIX(W, W_grad, size) do { \
        for(int i = 0; i < size; i++) { \
            W[i] = W[i] * (1.0f - learning_rate * ssm->weight_decay) - \
                   learning_rate * W_grad[i] / ssm->batch_size; \
        } \
    } while(0)
    
    UPDATE_MATRIX(ssm->A, ssm->A_grad, ssm->state_dim * ssm->state_dim);
    UPDATE_MATRIX(ssm->B, ssm->B_grad, ssm->state_dim * ssm->input_dim);
    UPDATE_MATRIX(ssm->C, ssm->C_grad, ssm->output_dim * ssm->state_dim);
    UPDATE_MATRIX(ssm->D, ssm->D_grad, ssm->output_dim * ssm->input_dim);
    
    #undef UPDATE_MATRIX
}

void save_model(SSM* ssm, const char* filename) {
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
    
    // Save FIM matrices
    fwrite(ssm->A_fim, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B_fim, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C_fim, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D_fim, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    
    fclose(file);
}

SSM* load_model(const char* filename) {
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
    
    // Load FIM matrices
    fread(ssm->A_fim, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B_fim, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C_fim, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D_fim, sizeof(float), output_dim * input_dim, file);
    
    return ssm;
}

#endif