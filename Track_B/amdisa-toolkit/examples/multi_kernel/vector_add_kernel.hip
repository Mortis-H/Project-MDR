// vector_add_kernel.hip
// Device-only kernel file for vector addition

#include <hip/hip_runtime.h>

extern "C" __global__ void vectorAdd(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 更複雜的範例：帶 shared memory 的 kernel
extern "C" __global__ void vectorAddShared(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ c,
                                            int n) {
    __shared__ float shared_a[256];
    __shared__ float shared_b[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 載入到 shared memory
    if (idx < n) {
        shared_a[tid] = a[idx];
        shared_b[tid] = b[idx];
    }
    __syncthreads();
    
    // 計算並寫回
    if (idx < n) {
        c[idx] = shared_a[tid] + shared_b[tid];
    }
}

