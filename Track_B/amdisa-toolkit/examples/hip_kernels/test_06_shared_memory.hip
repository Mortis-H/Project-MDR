// Level 3: 共享記憶體測試
// 測試 LDS (ds_read/ds_write) 指令

#include <hip/hip_runtime.h>

// 使用 shared memory 的簡單 reduction
__global__ void sharedMemKernel(const int* input, 
                                int* output,
                                int n) {
    __shared__ int shared[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load 到 shared memory (ds_write)
    shared[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // 簡單的 reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // 寫回結果
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(int);
    
    int* h_input = new int[N];
    int* h_output = new int[4];  // 4 blocks
    
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;  // 每個元素為 1
    }
    
    int *d_input, *d_output;
    hipMalloc(&d_input, bytes);
    hipMalloc(&d_output, 4 * sizeof(int));
    
    hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(sharedMemKernel, dim3(4), dim3(256), 
                       0, 0, d_input, d_output, N);
    
    hipMemcpy(h_output, d_output, 4 * sizeof(int), hipMemcpyDeviceToHost);
    
    // 驗證：每個 block 應該有 256 個 1 的總和 = 256
    bool success = true;
    for (int i = 0; i < 4; i++) {
        if (h_output[i] != 256) {
            success = false;
            break;
        }
    }
    
    delete[] h_input;
    delete[] h_output;
    hipFree(d_input);
    hipFree(d_output);
    
    return success ? 0 : 1;
}

