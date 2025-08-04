#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // State space matrices
    float* A;           // state_dim x state_dim (state transition) - deprecated, kept for compatibility
    float* B;           // state_dim x input_dim (input to state)
    float* C;           // output_dim x state_dim (state to output)
    float* D;           // output_dim x input_dim (input to output)
    
    // Orthogonal parameterization using Givens rotations
    float* rotation_angles;     // n(n-1)/2 rotation angles for Givens rotations
    float* A_orthogonal;        // state_dim x state_dim orthogonal matrix constructed from angles
    
    // Gradients
    float* A_grad;              // state_dim x state_dim - deprecated, kept for compatibility
    float* rotation_angles_grad;// n(n-1)/2 gradients for rotation angles
    float* B_grad;              // state_dim x input_dim
    float* C_grad;              // output_dim x state_dim
    float* D_grad;              // output_dim x input_dim
    
    // Adam parameters for rotation_angles, B, C, D
    float* rotation_angles_m; float* rotation_angles_v;
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
    
    // Temporary buffers for gradient computation (to avoid malloc/free in backward pass)
    float* A_temp;          // state_dim x state_dim temporary matrix for gradient computation
    
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
void build_orthogonal_from_angles(SSM* ssm);
void apply_givens_rotation(float* matrix, int n, int i, int j, float cos_theta, float sin_theta);
void forward_pass_ssm(SSM* ssm, float* X_t, int timestep);
float calculate_loss_ssm(SSM* ssm, float* y);
void zero_gradients_ssm(SSM* ssm);
void backward_pass_ssm(SSM* ssm, float* X);
void update_weights_ssm(SSM* ssm, float learning_rate);
void save_ssm(SSM* ssm, const char* filename);
SSM* load_ssm(const char* filename, int custom_batch_size);

#endif