// Level 2: 循環測試
// 測試 loop 控制流（backward branch）

#include <hip/hip_runtime.h>

// 簡單的累加循環
__global__ void loopKernel(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int sum = 0;
        
        // 簡單循環 (會產生 label + backward branch)
        for (int i = 0; i < 10; i++) {
            sum += i;
        }
        
        output[idx] = sum;
    }
}

int main() {
    const int N = 256;
    const size_t bytes = N * sizeof(int);
    
    int* h_output = new int[N];
    int *d_output;
    hipMalloc(&d_output, bytes);
    
    int threadsPerBlock = 256;
    hipLaunchKernelGGL(loopKernel, dim3(1), dim3(threadsPerBlock), 
                       0, 0, d_output, N);
    
    hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost);
    
    // 驗證：sum(0..9) = 45
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != 45) {
            success = false;
            break;
        }
    }
    
    delete[] h_output;
    hipFree(d_output);
    
    return success ? 0 : 1;
}

