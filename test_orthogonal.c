#include "ssm.h"
#include <stdio.h>

void print_matrix(const char* name, float* matrix, int n) {
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

float compute_matrix_norm(float* matrix, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float val = matrix[i * n + j];
            norm += val * val;
        }
    }
    return sqrtf(norm);
}

int main() {
    printf("Testing orthogonal parameterization...\n\n");
    
    // Create a small SSM to test
    int state_dim = 4;
    SSM* ssm = init_ssm(2, state_dim, 2, 10, 4);
    
    printf("A_skew parameters: ");
    int skew_params = state_dim * (state_dim - 1) / 2;
    for (int i = 0; i < skew_params; i++) {
        printf("%.4f ", ssm->A_skew[i]);
    }
    printf("\n\n");
    
    // Print the skew-symmetric matrix
    float* A_skew_full = (float*)malloc(state_dim * state_dim * sizeof(float));
    create_skew_symmetric(A_skew_full, ssm->A_skew, state_dim);
    print_matrix("A_skew (full matrix)", A_skew_full, state_dim);
    
    // Print the orthogonal matrix
    print_matrix("A_orthogonal", ssm->A_orthogonal, state_dim);
    
    // Test orthogonality: A^T * A should be identity
    float* AT_A = (float*)malloc(state_dim * state_dim * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                state_dim, state_dim, state_dim,
                1.0f, ssm->A_orthogonal, state_dim,
                ssm->A_orthogonal, state_dim,
                0.0f, AT_A, state_dim);
    
    print_matrix("A_orthogonal^T * A_orthogonal (should be identity)", AT_A, state_dim);
    
    // Compute spectral norm (should be close to 1 for orthogonal matrix)
    float norm = compute_matrix_norm(ssm->A_orthogonal, state_dim);
    printf("Frobenius norm of A_orthogonal: %.6f\n", norm);
    printf("Expected for orthogonal matrix: %.6f\n", sqrtf(state_dim));
    
    // Check if AT_A is close to identity
    float identity_error = 0.0f;
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float error = fabsf(AT_A[i * state_dim + j] - expected);
            identity_error += error;
        }
    }
    
    printf("Average error from identity: %.8f\n", identity_error / (state_dim * state_dim));
    printf("Test %s\n", (identity_error / (state_dim * state_dim) < 0.01f) ? "PASSED" : "FAILED");
    
    free(A_skew_full);
    free(AT_A);
    free_ssm(ssm);
    
    return 0;
}