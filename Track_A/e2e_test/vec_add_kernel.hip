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

        // if (i == 3) {
        //   print(A)
        //   print(B)
        // }

        C[i] = A[i] + B[i];

        // if (i == 2) print(C)

        // if (i < N) {
        //   print(A)
        //   print(B)
        //   print(C)
        // }

        // print(B*B-A*C*4)
    }
}

