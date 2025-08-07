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
    float* W_B;         // state_dim x (input_dim * input_dim) (weights for input-dependent B)
    float* C;           // output_dim x state_dim (state to output)
    float* D;           // output_dim x input_dim (input to output)
    
    // Gradients
    float* A_grad;      // state_dim x state_dim
    float* W_B_grad;    // state_dim x (input_dim * input_dim)
    float* C_grad;      // output_dim x state_dim
    float* D_grad;      // output_dim x input_dim
    
    // Adam parameters for A, W_B, C, D
    float* A_m; float* A_v;
    float* W_B_m; float* W_B_v;
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
    float* X_outer;        // seq_len x batch_size x (input_dim * input_dim) (outer products)
    
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

#endif