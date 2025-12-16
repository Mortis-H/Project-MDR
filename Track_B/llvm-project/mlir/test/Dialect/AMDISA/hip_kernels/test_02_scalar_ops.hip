// Level 1: Scalar 指令操作測試
// 測試 s_* 指令（scalar ALU operations）

#include <hip/hip_runtime.h>

// 測試 scalar 運算的 kernel
__global__ void scalarOps(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 這些操作會編譯成 scalar 指令
        int a = idx;
        int b = idx + 1;
        
        // 基本算術
        int sum = a + b;           // s_add_i32
        int diff = a - b;          // s_sub_i32
        int prod = a * b;          // s_mul_i32
        
        // 位運算
        int and_result = a & b;    // s_and_b32
        int or_result = a | b;     // s_or_b32
        int xor_result = a ^ b;    // s_xor_b32
        
        // 移位運算
        int lshift = a << 2;       // s_lshl_b32
        int rshift = a >> 2;       // s_ashr_i32
        
        // 組合結果
        output[idx] = sum + diff + prod + and_result + 
                      or_result + xor_result + lshift + rshift;
    }
}

int main() {
    const int N = 256;
    const size_t bytes = N * sizeof(int);
    
    int* h_output = new int[N];
    int *d_output;
    hipMalloc(&d_output, bytes);
    
    hipLaunchKernelGGL(scalarOps, dim3(1), dim3(N), 0, 0, d_output, N);
    
    hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost);
    
    // 簡單驗證：檢查第一個元素
    bool success = (h_output[0] != 0);
    
    delete[] h_output;
    hipFree(d_output);
    
    return success ? 0 : 1;
}

