#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLAS Error checking macro
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Device pointers for state space matrices
    float* d_A;           // state_dim x state_dim (state transition)
    float* d_B;           // state_dim x input_dim (input to state)
    float* d_C;           // output_dim x state_dim (state to output)
    float* d_D;           // output_dim x input_dim (input to output)
    
    // Device pointers for gradients
    float* d_A_grad;      // state_dim x state_dim
    float* d_B_grad;      // state_dim x input_dim
    float* d_C_grad;      // output_dim x state_dim
    float* d_D_grad;      // output_dim x input_dim
    
    // Device pointers for Adam parameters
    float* d_A_m; float* d_A_v;
    float* d_B_m; float* d_B_v;
    float* d_C_m; float* d_C_v;
    float* d_D_m; float* d_D_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Device pointers for helper arrays (time-major format)
    float* d_states;          // seq_len x batch_size x state_dim
    float* d_predictions;     // seq_len x batch_size x output_dim
    float* d_error;          // seq_len x batch_size x output_dim
    float* d_state_error;    // seq_len x batch_size x state_dim
    float* d_state_outputs;  // seq_len x batch_size x state_dim
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} SSM;

// CUDA kernel prototypes
__global__ void swish_forward_kernel_ssm(float* output, float* input, int size);
__global__ void swish_backward_kernel_ssm(float* grad_input, float* grad_output, float* input, int size);
__global__ void calc_error_kernel_ssm(float* error, float* predictions, float* y, int size);
__global__ void adamw_update_kernel_ssm(float* weight, float* grad, float* m, float* v, float beta1, float beta2, float epsilon, float learning_rate, float weight_decay, float alpha_t, int size, int batch_size);

// Function prototypes
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size);
void free_ssm(SSM* ssm);
void forward_pass_ssm(SSM* ssm, float* d_X);
float calculate_loss_ssm(SSM* ssm, float* d_y);
void zero_gradients_ssm(SSM* ssm);
void backward_pass_ssm(SSM* ssm, float* d_X);
void update_weights_ssm(SSM* ssm, float learning_rate);
void save_ssm(SSM* ssm, const char* filename);
SSM* load_ssm(const char* filename, int custom_batch_size);

#endif