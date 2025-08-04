#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

// LAPACK function declaration
extern void sgesv_(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);

typedef struct {
    // State space matrices
    float* A_skew;      // n(n-1)/2 skew-symmetric parameters for A
    float* A_orthogonal; // state_dim x state_dim orthogonal A computed via Cayley transform
    float* B;           // state_dim x input_dim (input to state)
    float* C;           // output_dim x state_dim (state to output)
    float* D;           // output_dim x input_dim (input to output)
    
    // Gradients
    float* A_skew_grad; // n(n-1)/2 gradients w.r.t. skew-symmetric parameters
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
    
    // Pre-allocated workspace for Cayley transform (to avoid malloc/free in forward/backward)
    float* workspace_S;         // state_dim x state_dim matrix
    float* workspace_I_plus_S;  // state_dim x state_dim matrix  
    float* workspace_I_minus_S; // state_dim x state_dim matrix
    int* workspace_ipiv;        // state_dim pivot array
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} SSM;

// Function prototypes
void cayley_transform(float* A_skew, float* A_orthogonal, int state_dim,
                     float* workspace_S, float* workspace_I_plus_S, float* workspace_I_minus_S, int* workspace_ipiv);
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