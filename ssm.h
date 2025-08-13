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
    
    // Lion parameters
    float* A_m;        // Momentum for A
    float* B_m;        // Momentum for B
    float* C_m;        // Momentum for C
    float* D_m;        // Momentum for D
    float beta1;       // Momentum coefficient
    float beta2;       // Weight decay coefficient
    float weight_decay; // Weight decay parameter for Lion
    
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