// vec_add_kernel.hip
#include <hip/hip_runtime.h>

extern "C"
__global__ void vec_add(const float* A,
                        const float* B,
                        float*       C,
                        int          N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

