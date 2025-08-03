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
    float* B;           // state_dim x input_dim (input to state) - FIXED for backward compatibility
    float* C;           // output_dim x state_dim (state to output) - FIXED for backward compatibility
    float* D;           // output_dim x input_dim (input to output)
    
    // Input-dependent projection parameters for selective SSM
    float* W_B;         // state_dim x input_dim x input_dim (weights for B projection)
    float* W_C;         // output_dim x state_dim x input_dim (weights for C projection)
    
    // Gradients
    float* A_grad;      // state_dim x state_dim
    float* B_grad;      // state_dim x input_dim (unused in selective mode)
    float* C_grad;      // output_dim x state_dim (unused in selective mode)
    float* D_grad;      // output_dim x input_dim
    float* W_B_grad;    // input_dim x state_dim x input_dim
    float* W_C_grad;    // input_dim x output_dim x state_dim
    
    // Adam parameters for A, B, C, D (B, C unused in selective mode)
    float* A_m; float* A_v;
    float* B_m; float* B_v;
    float* C_m; float* C_v;
    float* D_m; float* D_v;
    
    // Adam parameters for selective SSM projection parameters
    float* W_B_m; float* W_B_v;
    float* W_C_m; float* W_C_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Helper arrays for forward/backward pass (time-major format)
    float* states;          // seq_len x batch_size x state_dim
    float* predictions;     // seq_len x batch_size x output_dim
    float* error;          // seq_len x batch_size x output_dim
    float* state_error;    // seq_len x batch_size x state_dim
    float* state_outputs;  // seq_len x batch_size x state_dim
    
    // Temporary matrices for input-dependent projections
    float* B_t;            // batch_size x state_dim x input_dim (current timestep B)
    float* C_t;            // batch_size x output_dim x state_dim (current timestep C)
    
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