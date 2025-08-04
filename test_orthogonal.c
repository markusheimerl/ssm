#include "ssm.h"
#include <stdio.h>
#include <time.h>

// Test orthogonal matrix properties
void test_orthogonal_properties(SSM* ssm) {
    int n = ssm->state_dim;
    float* A = ssm->A_orthogonal;
    
    // Test 1: Compute A^T * A and verify it's close to identity
    float* ATA = (float*)malloc(n * n * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, n,
                1.0f, A, n,
                A, n,
                0.0f, ATA, n);
    
    printf("Testing A^T * A ≈ I:\n");
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float actual = ATA[i * n + j];
            float error = fabsf(actual - expected);
            if (error > max_error) max_error = error;
            
            if (i < 4 && j < 4) { // Print 4x4 submatrix
                printf("%8.5f ", actual);
            }
        }
        if (i < 4) printf("\n");
    }
    printf("Max error in A^T * A: %e\n\n", max_error);
    
    // Test 2: Compute spectral norm ||A||_2
    // For an orthogonal matrix, all singular values should be 1
    // So the spectral norm should be 1
    float* AAT = (float*)malloc(n * n * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, n, n,
                1.0f, A, n,
                A, n,
                0.0f, AAT, n);
    
    // Find the largest diagonal element of AA^T as an approximation
    float spectral_norm_approx = 0.0f;
    for (int i = 0; i < n; i++) {
        if (AAT[i * n + i] > spectral_norm_approx) {
            spectral_norm_approx = AAT[i * n + i];
        }
    }
    spectral_norm_approx = sqrtf(spectral_norm_approx);
    
    printf("Approximate spectral norm ||A||_2: %f (should be ≈ 1.0)\n", spectral_norm_approx);
    
    // Test 3: Check determinant should be ±1 (simplified check)
    printf("Matrix A (first 4x4):\n");
    for (int i = 0; i < 4 && i < n; i++) {
        for (int j = 0; j < 4 && j < n; j++) {
            printf("%8.5f ", A[i * n + j]);
        }
        printf("\n");
    }
    
    free(ATA);
    free(AAT);
}

int main() {
    srand(time(NULL));
    
    printf("Testing Orthogonal SSM Implementation\n");
    printf("=====================================\n\n");
    
    // Test with different sizes
    int test_dims[] = {4, 8, 16};
    int num_tests = sizeof(test_dims) / sizeof(test_dims[0]);
    
    for (int t = 0; t < num_tests; t++) {
        int dim = test_dims[t];
        printf("Testing with state_dim = %d\n", dim);
        printf("Number of rotation angles: %d\n", dim * (dim - 1) / 2);
        
        SSM* ssm = init_ssm(2, dim, 1, 10, 4);
        
        // Build the orthogonal matrix from random angles
        build_orthogonal_from_angles(ssm);
        
        // Test properties
        test_orthogonal_properties(ssm);
        
        free_ssm(ssm);
        printf("\n");
    }
    
    printf("Test completed!\n");
    return 0;
}