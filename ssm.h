#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// BLAS constants and function declarations (since cblas.h is not available)
typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;

// Simple BLAS replacement for matrix multiplication
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                int m, int n, int k, float alpha, const float* A, int lda,
                const float* B, int ldb, float beta, float* C, int ldc);

typedef struct {
    // State space matrices
    float* A_skew;      // state_dim*(state_dim-1)/2 (skew-symmetric parameters)
    float* A_orthogonal; // state_dim x state_dim (computed from A_skew)
    float* B;           // state_dim x input_dim (input to state)
    float* C;           // output_dim x state_dim (state to output)
    float* D;           // output_dim x input_dim (input to output)
    
    // Gradients
    float* A_skew_grad; // state_dim*(state_dim-1)/2
    float* B_grad;      // state_dim x input_dim
    float* C_grad;      // output_dim x state_dim
    float* D_grad;      // output_dim x input_dim
    
    // Adam parameters for A_skew, B, C, D
    float* A_skew_m; float* A_skew_v;
    float* B_m; float* B_v;
    float* C_m; float* C_v;
    float* D_m; float* D_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Helper arrays for forward/backward pass (time-major format)
    float* states;          // seq_len x batch_size x state_dim
    float* predictions;     // seq_len x batch_size x output_dim
    float* error;          // seq_len x batch_size x output_dim
    float* state_error;    // seq_len x batch_size x state_dim
    float* state_outputs;  // seq_len x batch_size x state_dim
    
    // Workspace memory for matrix operations (to avoid malloc in forward/backward pass)
    float* A_skew_full;     // state_dim x state_dim workspace
    float* workspace;       // 5 * state_dim * state_dim workspace for matrix exponential
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} SSM;

// Function prototypes
void create_skew_symmetric(float* A_skew_full, const float* A_skew_params, int n);
void matrix_exponential_pade(float* exp_A, const float* A_skew_full, int n, float* workspace);
void matrix_exponential_gradient(float* grad_A_skew, const float* grad_exp_A, 
                                const float* A_skew_full, int n);

SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size);
void free_ssm(SSM* ssm);
void reset_state_ssm(SSM* ssm);
void forward_pass_ssm(SSM* ssm, float* X_t, int timestep);
float calculate_loss_ssm(SSM* ssm, float* y);
void zero_gradients_ssm(SSM* ssm);
void backward_pass_ssm(SSM* ssm, float* X);
void update_weights_ssm(SSM* ssm, float learning_rate);
void save_ssm(SSM* ssm, const char* filename);
SSM* load_ssm(const char* filename, int custom_batch_size);

#endif