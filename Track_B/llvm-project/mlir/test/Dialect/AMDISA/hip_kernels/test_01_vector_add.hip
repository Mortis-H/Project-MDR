// Level 1: 最簡單的向量加法測試
// 測試基本的 global memory load/store 和算術運算

#include <hip/hip_runtime.h>

// 簡單的向量加法 kernel
__global__ void vectorAdd(const float* __restrict__ a,
                          const float* __restrict__ b, 
                          float* __restrict__ c,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host 端驗證函數
extern "C" bool verify_vector_add(const float* a, const float* b, 
                                   const float* c, int n) {
    for (int i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > 1e-5) {
            return false;
        }
    }
    return true;
}

// Host 測試主函數
int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    
    // 分配 host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    
    // 初始化數據
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // 分配 device memory
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
    
    // 複製數據到 device
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    
    // 啟動 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 
                       0, 0, d_a, d_b, d_c, N);
    
    // 複製結果回 host
    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);
    
    // 驗證結果
    bool success = verify_vector_add(h_a, h_b, h_c, N);
    
    // 清理
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    
    return success ? 0 : 1;
}

