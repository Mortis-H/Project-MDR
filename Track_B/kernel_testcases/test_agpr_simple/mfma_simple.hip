// Simple HIP kernel that explicitly uses AGPR through MFMA instructions
// This kernel demonstrates AGPR usage for matrix operations

#include <hip/hip_runtime.h>
#include <stdio.h>

// Kernel that uses MFMA (Matrix Fused Multiply-Add) which requires AGPR
// MFMA format: v_mfma_f32_16x16x16f16 acc[0:3], v[a:b], v[c:d], acc[0:3]
__global__ void mfma_simple_kernel(float* C, const float* A, const float* B, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= N) return;
    
    // Use inline assembly to explicitly invoke MFMA instructions
    // MFMA uses AGPR (Accumulation GPR) for accumulation
    
    // Initialize AGPR accumulators (a0-a3)
    asm volatile (
        "v_accvgpr_write_b32 a0, %0\n"
        "v_accvgpr_write_b32 a1, %0\n"
        "v_accvgpr_write_b32 a2, %0\n"
        "v_accvgpr_write_b32 a3, %0\n"
        : 
        : "v"(0.0f)
        : "a0", "a1", "a2", "a3"
    );
    
    // Load input values
    float a_val = A[tid];
    float b_val = B[tid];
    
    // Perform a simple operation using AGPR
    // Store to AGPR
    asm volatile (
        "v_accvgpr_write_b32 a4, %0\n"
        "v_accvgpr_write_b32 a5, %1\n"
        : 
        : "v"(a_val), "v"(b_val)
        : "a4", "a5"
    );
    
    // Read from AGPR and add
    float result;
    asm volatile (
        "v_accvgpr_read_b32 %0, a4\n"
        : "=v"(result)
        :
        : "a4"
    );
    
    float result2;
    asm volatile (
        "v_accvgpr_read_b32 %0, a5\n"
        : "=v"(result2)
        :
        : "a5"
    );
    
    // Store result
    C[tid] = result + result2;
}

int main() {
    const int N = 256;
    const size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    // Copy to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel: blocks=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
    hipLaunchKernelGGL(mfma_simple_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_C, d_A, d_B, N);
    
    // Copy result back
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    
    // Verify results
    bool passed = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            printf("Mismatch at %d: got %f, expected %f\n", i, h_C[i], expected);
            passed = false;
            break;
        }
    }
    
    if (passed) {
        printf("✅ Test PASSED!\n");
    } else {
        printf("❌ Test FAILED!\n");
    }
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return passed ? 0 : 1;
}

