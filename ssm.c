#include "ssm.h"

// Initialize the network with configurable dimensions
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
    ssm->weight_decay = 0.01f;
    
    // Allocate and initialize matrices and gradients
    ssm->A = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
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
    
    // Allocate layer outputs and working buffers
    ssm->layer1_preact = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->layer1_output = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->layer2_output = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    ssm->error_hidden = (float*)malloc(seq_len * batch_size * state_dim * sizeof(float));
    ssm->error_output = (float*)malloc(seq_len * batch_size * output_dim * sizeof(float));
    
    // Initialize matrices
    float scale_A = 0.5f / sqrtf(state_dim);
    float scale_B = 1.0f / sqrtf(input_dim);
    float scale_C = 1.0f / sqrtf(state_dim);
    float scale_D = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->A[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_A;
    }
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->B[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->C[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->D[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_D;
    }
    
    return ssm;
}

// Free network memory
void free_ssm(SSM* ssm) {
    free(ssm->A); free(ssm->B); free(ssm->C); free(ssm->D);
    free(ssm->A_grad); free(ssm->B_grad); free(ssm->C_grad); free(ssm->D_grad);
    free(ssm->A_m); free(ssm->A_v);
    free(ssm->B_m); free(ssm->B_v);
    free(ssm->C_m); free(ssm->C_v);
    free(ssm->D_m); free(ssm->D_v);
    free(ssm->layer1_preact); free(ssm->layer1_output); free(ssm->layer2_output);
    free(ssm->error_output); free(ssm->error_hidden);
    free(ssm);
}

// Reset state for new sequence
void reset_state_ssm(SSM* ssm) {
    memset(ssm->layer1_preact, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    memset(ssm->layer1_output, 0, ssm->seq_len * ssm->batch_size * ssm->state_dim * sizeof(float));
    memset(ssm->layer2_output, 0, ssm->seq_len * ssm->batch_size * ssm->output_dim * sizeof(float));
}

// Forward pass for single timestep
void forward_pass_ssm(SSM* ssm, float* X_t, int timestep) {
    // Get pointers to current timestep data (time-major format)
    float* Z_t = &ssm->layer1_preact[timestep * ssm->batch_size * ssm->state_dim];
    float* H_t = &ssm->layer1_output[timestep * ssm->batch_size * ssm->state_dim];
    float* Y_t = &ssm->layer2_output[timestep * ssm->batch_size * ssm->output_dim];
    
    // Z_t = X_t B^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim, ssm->input_dim,
                1.0f, X_t, ssm->input_dim,
                ssm->B, ssm->input_dim,
                0.0f, Z_t, ssm->state_dim);
    
    // Z_t = Z_t + Z_{t-1} A^T
    if (timestep > 0) {
        float* Z_prev = &ssm->layer1_preact[(timestep-1) * ssm->batch_size * ssm->state_dim];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    ssm->batch_size, ssm->state_dim, ssm->state_dim,
                    1.0f, Z_prev, ssm->state_dim,
                    ssm->A, ssm->state_dim,
                    1.0f, Z_t, ssm->state_dim);
    }
    
    // H_t = Z_t * swish(Z_t)
    for (int i = 0; i < ssm->batch_size * ssm->state_dim; i++) {
        H_t[i] = Z_t[i] / (1.0f + expf(-Z_t[i]));
    }
    
    // Y_t = H_t C^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->output_dim, ssm->state_dim,
                1.0f, H_t, ssm->state_dim,
                ssm->C, ssm->state_dim,
                0.0f, Y_t, ssm->output_dim);
    
    // Y_t = Y_t + X_t D^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->output_dim, ssm->input_dim,
                1.0f, X_t, ssm->input_dim,
                ssm->D, ssm->input_dim,
                1.0f, Y_t, ssm->output_dim);
}

// Calculate loss
float calculate_loss_ssm(SSM* ssm, float* y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;
    for (int i = 0; i < ssm->seq_len * ssm->batch_size * ssm->output_dim; i++) {
        ssm->error_output[i] = ssm->layer2_output[i] - y[i];
        loss += ssm->error_output[i] * ssm->error_output[i];
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

// Backward pass for single timestep
void backward_pass_ssm(SSM* ssm, float* X_t, int timestep) {
    // Get pointers to current timestep data
    float* Z_t = &ssm->layer1_preact[timestep * ssm->batch_size * ssm->state_dim];
    float* H_t = &ssm->layer1_output[timestep * ssm->batch_size * ssm->state_dim];
    float* error_output_t = &ssm->error_output[timestep * ssm->batch_size * ssm->output_dim];
    float* error_hidden_t = &ssm->error_hidden[timestep * ssm->batch_size * ssm->state_dim];
    
    // ∂L/∂C += (∂L/∂Y_t)^T H_t
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->output_dim, ssm->state_dim, ssm->batch_size,
                1.0f, error_output_t, ssm->output_dim,
                H_t, ssm->state_dim,
                1.0f, ssm->C_grad, ssm->state_dim);
    
    // ∂L/∂D += (∂L/∂Y_t)^T X_t
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->output_dim, ssm->input_dim, ssm->batch_size,
                1.0f, error_output_t, ssm->output_dim,
                X_t, ssm->input_dim,
                1.0f, ssm->D_grad, ssm->input_dim);
    
    // ∂L/∂H_t = (∂L/∂Y_t) C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->state_dim, ssm->output_dim,
                1.0f, error_output_t, ssm->output_dim,
                ssm->C, ssm->state_dim,
                0.0f, error_hidden_t, ssm->state_dim);
    
    // ∂L/∂Z_t = ∂L/∂H_t ⊙ [σ(Z_t) + Z_t σ(Z_t)(1-σ(Z_t))]
    for (int i = 0; i < ssm->batch_size * ssm->state_dim; i++) {
        float z = Z_t[i];
        float sigmoid = 1.0f / (1.0f + expf(-z));
        error_hidden_t[i] *= sigmoid + z * sigmoid * (1.0f - sigmoid);
    }
    
    // ∂L/∂B += (∂L/∂Z_t)^T X_t
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->state_dim, ssm->input_dim, ssm->batch_size,
                1.0f, error_hidden_t, ssm->state_dim,
                X_t, ssm->input_dim,
                1.0f, ssm->B_grad, ssm->input_dim);
    
    // Propagate error to previous timestep
    if (timestep > 0) {
        float* Z_prev = &ssm->layer1_preact[(timestep-1) * ssm->batch_size * ssm->state_dim];
        float* error_hidden_prev = &ssm->error_hidden[(timestep-1) * ssm->batch_size * ssm->state_dim];
        
        // ∂L/∂A += (∂L/∂Z_t)^T Z_{t-1}
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    ssm->state_dim, ssm->state_dim, ssm->batch_size,
                    1.0f, error_hidden_t, ssm->state_dim,
                    Z_prev, ssm->state_dim,
                    1.0f, ssm->A_grad, ssm->state_dim);
        
        // ∂L/∂Z_{t-1} += (∂L/∂Z_t) A
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    ssm->batch_size, ssm->state_dim, ssm->state_dim,
                    1.0f, error_hidden_t, ssm->state_dim,
                    ssm->A, ssm->state_dim,
                    1.0f, error_hidden_prev, ssm->state_dim);
    }
}

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate) {
    ssm->t++;  // Increment time step
    
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int total_samples = ssm->seq_len * ssm->batch_size;
    
    // Update A weights
    for (int i = 0; i < ssm->state_dim * ssm->state_dim; i++) {
        float grad = ssm->A_grad[i] / total_samples;
        
        ssm->A_m[i] = ssm->beta1 * ssm->A_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->A_v[i] = ssm->beta2 * ssm->A_v[i] + (1.0f - ssm->beta2) * grad * grad;
        
        float update = alpha_t * ssm->A_m[i] / (sqrtf(ssm->A_v[i]) + ssm->epsilon);
        ssm->A[i] = ssm->A[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update B weights
    for (int i = 0; i < ssm->state_dim * ssm->input_dim; i++) {
        float grad = ssm->B_grad[i] / total_samples;
        
        ssm->B_m[i] = ssm->beta1 * ssm->B_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->B_v[i] = ssm->beta2 * ssm->B_v[i] + (1.0f - ssm->beta2) * grad * grad;
        
        float update = alpha_t * ssm->B_m[i] / (sqrtf(ssm->B_v[i]) + ssm->epsilon);
        ssm->B[i] = ssm->B[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update C weights
    for (int i = 0; i < ssm->output_dim * ssm->state_dim; i++) {
        float grad = ssm->C_grad[i] / total_samples;
        
        ssm->C_m[i] = ssm->beta1 * ssm->C_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->C_v[i] = ssm->beta2 * ssm->C_v[i] + (1.0f - ssm->beta2) * grad * grad;
        
        float update = alpha_t * ssm->C_m[i] / (sqrtf(ssm->C_v[i]) + ssm->epsilon);
        ssm->C[i] = ssm->C[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
    
    // Update D weights
    for (int i = 0; i < ssm->output_dim * ssm->input_dim; i++) {
        float grad = ssm->D_grad[i] / total_samples;
        
        ssm->D_m[i] = ssm->beta1 * ssm->D_m[i] + (1.0f - ssm->beta1) * grad;
        ssm->D_v[i] = ssm->beta2 * ssm->D_v[i] + (1.0f - ssm->beta2) * grad * grad;
        
        float update = alpha_t * ssm->D_m[i] / (sqrtf(ssm->D_v[i]) + ssm->epsilon);
        ssm->D[i] = ssm->D[i] * (1.0f - learning_rate * ssm->weight_decay) - update;
    }
}

// Function to save model weights to binary file
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

// Function to load model weights from binary file
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
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
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