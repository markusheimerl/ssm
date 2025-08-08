#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // State space matrices
    float* A;           // state_dim x state_dim
    float* B;           // state_dim x input_dim
    float* C;           // output_dim x state_dim
    float* D;           // output_dim x input_dim
    
    // Gradients
    float* A_grad;      // state_dim x state_dim
    float* B_grad;      // state_dim x input_dim
    float* C_grad;      // output_dim x state_dim
    float* D_grad;      // output_dim x input_dim
    
    // Adam parameters
    float* A_m;  // First moment for A
    float* A_v;  // Second moment for A
    float* B_m;  // First moment for B
    float* B_v;  // Second moment for B
    float* C_m;  // First moment for C
    float* C_v;  // Second moment for C
    float* D_m;  // First moment for D
    float* D_v;  // Second moment for D
    float beta1;   // Exponential decay rate for first moment
    float beta2;   // Exponential decay rate for second moment
    float epsilon; // Small constant for numerical stability
    int t;         // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Layer outputs and working buffers
    float* layer1_preact;   // seq_len x batch_size x state_dim
    float* layer1_output;   // seq_len x batch_size x state_dim
    float* layer2_output;   // seq_len x batch_size x output_dim
    float* error_hidden;    // seq_len x batch_size x state_dim
    float* error_output;    // seq_len x batch_size x output_dim
    
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
void backward_pass_ssm(SSM* ssm, float* X_t, int timestep);
void update_weights_ssm(SSM* ssm, float learning_rate);
void save_ssm(SSM* ssm, const char* filename);
SSM* load_ssm(const char* filename, int custom_batch_size);

#endif