#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "mlp/mlp.h"

typedef struct {
    // State space matrices
    float* A;           // state_dim x state_dim (state transition) - now unused, replaced by A_t
    float* B;           // state_dim x input_dim (input to state)
    float* C;           // output_dim x state_dim (state to output)
    float* D;           // output_dim x input_dim (input to output)
    
    // MLP parameters for input-dependent state transition
    float* W1;          // mlp_hidden_dim x input_dim (first MLP layer)
    float* W2;          // state_dim^2 x mlp_hidden_dim (second MLP layer, outputs A_t matrix)
    
    // Gradients
    float* A_grad;      // state_dim x state_dim - now unused
    float* B_grad;      // state_dim x input_dim
    float* C_grad;      // output_dim x state_dim
    float* D_grad;      // output_dim x input_dim
    float* W1_grad;     // mlp_hidden_dim x input_dim
    float* W2_grad;     // state_dim^2 x mlp_hidden_dim
    
    // Adam parameters for A, B, C, D, W1, W2
    float* A_m; float* A_v;  // unused but kept for compatibility
    float* B_m; float* B_v;
    float* C_m; float* C_v;
    float* D_m; float* D_v;
    float* W1_m; float* W1_v;
    float* W2_m; float* W2_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Helper arrays for forward/backward pass (time-major format)
    float* states;          // seq_len x batch_size x state_dim
    float* predictions;     // seq_len x batch_size x output_dim
    float* error;          // seq_len x batch_size x output_dim
    float* state_error;    // seq_len x batch_size x state_dim
    float* state_outputs;  // seq_len x batch_size x state_dim
    
    // MLP helper arrays
    float* Z_t;            // batch_size x mlp_hidden_dim (MLP first layer output)
    float* U_t;            // batch_size x mlp_hidden_dim (gated activation)
    float* A_t;            // batch_size x state_dim x state_dim (input-dependent A matrix)
    float* Z_error;        // batch_size x mlp_hidden_dim (error for Z_t)
    float* U_error;        // batch_size x mlp_hidden_dim (error for U_t)
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
    int mlp_hidden_dim;     // hidden dimension for MLP
} SSM;

// Function prototypes
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size, int mlp_hidden_dim);
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