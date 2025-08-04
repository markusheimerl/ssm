#include "ssm.h"

// Test if A_orthogonal is indeed orthogonal (A^T * A = I)
void test_orthogonal_property(float* A, int n) {
    printf("Testing orthogonal property of A matrix (%dx%d):\n", n, n);
    
    // Compute A^T * A
    float* AtA = (float*)malloc(n * n * sizeof(float));
    
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, n,
                1.0f, A, n,
                A, n,
                0.0f, AtA, n);
    
    // Check if AtA is close to identity
    float max_off_diagonal = 0.0f;
    float min_diagonal = 1.0f;
    float max_diagonal = 1.0f;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Diagonal elements should be 1
                min_diagonal = fminf(min_diagonal, AtA[i * n + j]);
                max_diagonal = fmaxf(max_diagonal, AtA[i * n + j]);
            } else {
                // Off-diagonal elements should be 0
                max_off_diagonal = fmaxf(max_off_diagonal, fabsf(AtA[i * n + j]));
            }
        }
    }
    
    printf("Diagonal range: [%.6f, %.6f] (should be close to 1.0)\n", min_diagonal, max_diagonal);
    printf("Max off-diagonal: %.6f (should be close to 0.0)\n", max_off_diagonal);
    
    // Compute spectral norm ||A||_2
    float spectral_norm = 0.0f;
    for (int i = 0; i < n; i++) {
        float row_norm = 0.0f;
        for (int j = 0; j < n; j++) {
            row_norm += A[i * n + j] * A[i * n + j];
        }
        spectral_norm = fmaxf(spectral_norm, sqrtf(row_norm));
    }
    printf("Approximate spectral norm ||A||_2: %.6f (should be close to 1.0)\n", spectral_norm);
    
    free(AtA);
}

int main() {
    printf("Testing Cayley Transform Orthogonal Property\n");
    printf("==========================================\n\n");
    
    // Test with different sizes
    int sizes[] = {4, 8, 16};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    srand(42); // Fixed seed for reproducibility
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        int skew_params = n * (n - 1) / 2;
        
        printf("Testing with state_dim = %d (skew_params = %d)\n", n, skew_params);
        
        // Create random skew-symmetric parameters
        float* A_skew = (float*)malloc(skew_params * sizeof(float));
        float* A_orthogonal = (float*)malloc(n * n * sizeof(float));
        
        float scale = 0.1f / sqrtf(n);
        for (int i = 0; i < skew_params; i++) {
            A_skew[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
        
        // Compute orthogonal matrix via Cayley transform
        cayley_transform(A_skew, A_orthogonal, n);
        
        // Test orthogonal property
        test_orthogonal_property(A_orthogonal, n);
        
        printf("\n");
        
        free(A_skew);
        free(A_orthogonal);
    }
    
    return 0;
}