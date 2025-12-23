// Level 1: 記憶體操作測試
// 測試各種 memory load/store 指令

#include <hip/hip_runtime.h>

// 測試不同的 memory access pattern
__global__ void memoryOps(const int* __restrict__ input,
                          int* __restrict__ output,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Global memory load (global_load_dword)
        int value = input[idx];
        
        // 簡單運算
        value = value * 2 + 1;
        
        // Global memory store (global_store_dword)
        output[idx] = value;
    }
}

int main() {
    const int N = 512;
    const size_t bytes = N * sizeof(int);
    
    int* h_input = new int[N];
    int* h_output = new int[N];
    
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }
    
    int *d_input, *d_output;
    hipMalloc(&d_input, bytes);
    hipMalloc(&d_output, bytes);
    
    hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(memoryOps, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_input, d_output, N);
    
    hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost);
    
    // 驗證
    bool success = true;
    for (int i = 0; i < N; i++) {
        int expected = h_input[i] * 2 + 1;
        if (h_output[i] != expected) {
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

