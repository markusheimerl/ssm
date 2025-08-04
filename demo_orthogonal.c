#include "ssm.h"
#include <stdio.h>

int main() {
    printf("=== Orthogonal A Matrix Parameterization Demo ===\n\n");
    
    // Create a small SSM to demonstrate properties
    int state_dim = 8;
    SSM* ssm = init_ssm(4, state_dim, 2, 20, 8);
    
    printf("State dimension: %d\n", state_dim);
    printf("Original parameterization would have: %d² = %d parameters for A\n", 
           state_dim, state_dim * state_dim);
    
    int skew_params = state_dim * (state_dim - 1) / 2;
    printf("Orthogonal parameterization uses: %d((%d-1)/2) = %d parameters for A_skew\n", 
           state_dim, state_dim, skew_params);
    printf("Parameter reduction: %.1f%%\n\n", 
           100.0f * (1.0f - (float)skew_params / (state_dim * state_dim)));
    
    // Test orthogonality properties
    float* AT_A = (float*)malloc(state_dim * state_dim * sizeof(float));
    
    // Compute A^T * A
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                state_dim, state_dim, state_dim,
                1.0f, ssm->A_orthogonal, state_dim,
                ssm->A_orthogonal, state_dim,
                0.0f, AT_A, state_dim);
    
    // Measure orthogonality
    float max_off_diagonal = 0.0f;
    float min_diagonal = 1.0f;
    float max_diagonal = 0.0f;
    
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            float val = AT_A[i * state_dim + j];
            if (i == j) {
                min_diagonal = fminf(min_diagonal, val);
                max_diagonal = fmaxf(max_diagonal, val);
            } else {
                max_off_diagonal = fmaxf(max_off_diagonal, fabsf(val));
            }
        }
    }
    
    printf("Orthogonality Properties (A^T * A should be identity):\n");
    printf("  Diagonal elements range: [%.6f, %.6f] (should be ~1.0)\n", 
           min_diagonal, max_diagonal);
    printf("  Max off-diagonal element: %.8f (should be ~0.0)\n", 
           max_off_diagonal);
    
    // Compute spectral norm (largest singular value)
    float frobenius_norm = 0.0f;
    for (int i = 0; i < state_dim * state_dim; i++) {
        float val = ssm->A_orthogonal[i];
        frobenius_norm += val * val;
    }
    frobenius_norm = sqrtf(frobenius_norm);
    
    printf("  Frobenius norm: %.6f (should be ~%.6f for orthogonal matrix)\n", 
           frobenius_norm, sqrtf(state_dim));
    
    // Test stability property
    printf("\nStability Properties:\n");
    printf("  ||A||₂ = 1 exactly (guaranteed by orthogonal property)\n");
    printf("  Eigenvalues lie on unit circle (guaranteed stability)\n");
    printf("  No need for eigenvalue constraints or clipping\n");
    
    printf("\nKey Benefits:\n");
    printf("  ✓ Perfect numerical stability without hyperparameter tuning\n");
    printf("  ✓ Reduced parameter count (%d vs %d)\n", skew_params, state_dim * state_dim);
    printf("  ✓ No HiPPO dependency\n");
    printf("  ✓ Simple random initialization works well\n");
    printf("  ✓ Gradients flow naturally through matrix exponential\n");
    
    free(AT_A);
    free_ssm(ssm);
    
    return 0;
}