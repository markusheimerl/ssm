#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    __half* d_A;           // state_dim x state_dim
    __half* d_B;           // state_dim x input_dim
    __half* d_C;           // output_dim x state_dim
    __half* d_D;           // output_dim x input_dim
    
    // Device pointers for gradients
    float* d_A_grad;       // state_dim x state_dim
    float* d_B_grad;       // state_dim x input_dim
    float* d_C_grad;       // output_dim x state_dim
    float* d_D_grad;       // output_dim x input_dim
    
    // Device pointers for Lion parameters
    float* d_A_m;          // Momentum for A
    float* d_B_m;          // Momentum for B
    float* d_C_m;          // Momentum for C
    float* d_D_m;          // Momentum for D
    float beta1;           // Momentum coefficient
    float beta2;           // EMA coefficient for momentum update
    float weight_decay;    // Weight decay parameter for Lion
    
    // Device pointers for layer outputs and working buffers
    __half* d_layer1_preact;   // seq_len x batch_size x state_dim
    __half* d_layer1_output;   // seq_len x batch_size x state_dim
    __half* d_layer2_output;   // seq_len x batch_size x output_dim
    __half* d_error_hidden;    // seq_len x batch_size x state_dim
    __half* d_error_output;    // seq_len x batch_size x output_dim
    
    // Device memory for loss computation
    float* d_loss;
    
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
__global__ void swish_forward_kernel_ssm(__half* output, __half* input, int size);
__global__ void swish_backward_kernel_ssm(__half* grad_input, __half* grad_output, __half* input, int size);
__global__ void lion_update_kernel_ssm(__half* weight, float* grad, float* m, float beta1, float beta2, float learning_rate, float weight_decay, int size, int total_samples);
__global__ void hgeam_kernel_ssm(__half* C, __half alpha, __half* A, __half beta, __half* B, int rows, int cols);
__global__ void hdot_kernel_ssm(__half* x, float* result, int size);

// Function prototypes
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size, cublasHandle_t cublas_handle);
void free_ssm(SSM* ssm);
void reset_state_ssm(SSM* ssm);
void forward_pass_ssm(SSM* ssm, __half* d_X_t, int timestep);
float calculate_loss_ssm(SSM* ssm, __half* d_y);
void zero_gradients_ssm(SSM* ssm);
void backward_pass_ssm(SSM* ssm, __half* d_X_t, int timestep);
void update_weights_ssm(SSM* ssm, float learning_rate);
void save_ssm(SSM* ssm, const char* filename);
SSM* load_ssm(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle);

#endif