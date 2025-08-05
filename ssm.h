#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // State space matrices
    float* A;           // state_dim x state_dim (computed from blocks)
    float* B;           // state_dim x input_dim (input to state)
    float* C;           // output_dim x state_dim (state to output)
    float* D;           // output_dim x input_dim (input to output)
    
    // Block-diagonal orthogonal parameterization
    int num_blocks;     // number of blocks
    int block_size;     // size of each block
    float* A_skew;      // flattened skew-symmetric parameters for all blocks
    int skew_params_per_block;  // block_size * (block_size - 1) / 2
    
    // Gradients
    float* A_grad;      // state_dim x state_dim
    float* B_grad;      // state_dim x input_dim
    float* C_grad;      // output_dim x state_dim
    float* D_grad;      // output_dim x input_dim
    float* A_skew_grad; // gradient w.r.t. skew parameters
    
    // Adam parameters for B, C, D, A_skew
    float* B_m; float* B_v;
    float* C_m; float* C_v;
    float* D_m; float* D_v;
    float* A_skew_m; float* A_skew_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Helper arrays for forward/backward pass (time-major format)
    float* states;          // seq_len x batch_size x state_dim
    float* predictions;     // seq_len x batch_size x output_dim
    float* error;          // seq_len x batch_size x output_dim
    float* state_error;    // seq_len x batch_size x state_dim
    float* state_outputs;  // seq_len x batch_size x state_dim
    
    // Working memory for matrix exponential computation
    float* block_work;     // workspace for individual block operations
    float* exp_blocks;     // temporary storage for exp(S) blocks
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} SSM;

// Function prototypes
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

// Block-diagonal orthogonal matrix utilities
void matrix_exp_pade(float* result, const float* skew_matrix, int size, float* workspace);
void construct_skew_matrix(float* skew_matrix, const float* params, int size);
void compute_A_from_blocks(SSM* ssm);
void compute_skew_gradients(SSM* ssm);

#endif