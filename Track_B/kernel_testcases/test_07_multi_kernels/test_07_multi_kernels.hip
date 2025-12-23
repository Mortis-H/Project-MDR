// Level 1: 多 Kernel 測試檔案
// 測試在同一個檔案中包含多個不同功能的 kernel

#include <hip/hip_runtime.h>
#include <cmath>

// Kernel 1: 向量加法
__global__ void vectorAdd(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel 2: 向量乘法
__global__ void vectorMul(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Kernel 3: 向量點積（使用 shared memory）
__global__ void vectorDot(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ partial_sums,
                          int n) {
    __shared__ float shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 載入數據並計算部分乘積
    float temp = 0.0f;
    if (idx < n) {
        temp = a[idx] * b[idx];
    }
    shared_data[tid] = temp;
    __syncthreads();
    
    // Reduction 在 shared memory 中
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // 將每個 block 的結果寫回 global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

// Kernel 4: 向量縮放和偏移（SAXPY: y = a*x + y）
__global__ void saxpy(float a,
                      const float* __restrict__ x,
                      float* __restrict__ y,
                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// Kernel 5: 條件運算（包含分支）
__global__ void conditionalOps(const float* __restrict__ input,
                               float* __restrict__ output,
                               float threshold,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = input[idx];
        // 條件分支
        if (value > threshold) {
            output[idx] = value * 2.0f;
        } else if (value < -threshold) {
            output[idx] = value / 2.0f;
        } else {
            output[idx] = value;
        }
    }
}

// Host 端驗證函數
extern "C" bool verify_vector_add(const float* a, const float* b,
                                   const float* c, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5) {
            return false;
        }
    }
    return true;
}

extern "C" bool verify_vector_mul(const float* a, const float* b,
                                   const float* c, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(c[i] - (a[i] * b[i])) > 1e-5) {
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
    float* h_d = new float[N];
    
    // 初始化數據
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
        h_d[i] = static_cast<float>(i * 0.5);
    }
    
    // 分配 device memory
    float *d_a, *d_b, *d_c, *d_d, *d_partial;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
    hipMalloc(&d_d, bytes);
    
    // 複製數據到 device
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_d, h_d, bytes, hipMemcpyHostToDevice);
    
    // 設置執行配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 測試 1: 向量加法
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_a, d_b, d_c, N);
    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);
    bool test1_pass = verify_vector_add(h_a, h_b, h_c, N);
    
    // 測試 2: 向量乘法
    hipLaunchKernelGGL(vectorMul, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_a, d_b, d_c, N);
    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);
    bool test2_pass = verify_vector_mul(h_a, h_b, h_c, N);
    
    // 測試 3: 向量點積
    size_t partial_bytes = blocksPerGrid * sizeof(float);
    hipMalloc(&d_partial, partial_bytes);
    hipLaunchKernelGGL(vectorDot, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_a, d_b, d_partial, N);
    
    // 測試 4: SAXPY
    float alpha = 2.5f;
    hipLaunchKernelGGL(saxpy, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, alpha, d_a, d_d, N);
    
    // 測試 5: 條件運算
    float threshold = 500.0f;
    hipLaunchKernelGGL(conditionalOps, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_a, d_c, threshold, N);
    
    hipDeviceSynchronize();
    
    // 清理
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_d;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipFree(d_d);
    hipFree(d_partial);
    
    return (test1_pass && test2_pass) ? 0 : 1;
}

