// Level 2: 條件分支測試
// 測試 s_cbranch, s_branch 等控制流指令

#include <hip/hip_runtime.h>

// 測試條件分支
__global__ void conditionalKernel(const int* input, 
                                  int* output,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int value = input[idx];
        
        // 條件判斷 (會產生 v_cmp + s_cbranch)
        if (value % 2 == 0) {
            output[idx] = value * 2;
        } else {
            output[idx] = value * 3 + 1;
        }
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
    hipLaunchKernelGGL(conditionalKernel, dim3(blocksPerGrid), 
                       dim3(threadsPerBlock), 0, 0, d_input, d_output, N);
    
    hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost);
    
    // 驗證
    bool success = true;
    for (int i = 0; i < N; i++) {
        int expected = (h_input[i] % 2 == 0) ? 
                       h_input[i] * 2 : h_input[i] * 3 + 1;
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

