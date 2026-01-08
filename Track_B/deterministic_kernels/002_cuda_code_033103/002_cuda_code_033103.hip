#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define N 64
#define EPS 1e-8f
#define LR 0.001f
#define BETA1 0.9f
#define BETA2 0.999f

__global__ void adamUpdateKernel(float* params, float* grads, float* m, float* v, int step) {
    extern __shared__ float sharedData[];

    int idx = threadIdx.x;
    float g = grads[idx];
    float m_t = m[idx] * BETA1 + (1 - BETA1) * g;
    float v_t = v[idx] * BETA2 + (1 - BETA2) * g * g;
    float m_hat = m_t / (1 - powf(BETA1, step));
    float v_hat = v_t / (1 - powf(BETA2, step));
    float update = LR * m_hat / (sqrtf(v_hat) + EPS);
    params[idx] -= update;

    m[idx] = m_t;
    v[idx] = v_t;
}

__global__ void initKernel(float* array, float value) {
    int idx = threadIdx.x;
    array[idx] = value;
}

int main() {
    float* d_params, *d_grads, *d_m, *d_v;
    float* h_params = new float[N];
    float* h_grads = new float[N];

    // Initialize host parameters and gradients
    for (int i = 0; i < N; ++i) {
        h_params[i] = 0.0f;
        h_grads[i] = 1.0f / (i + 1); // Example gradient values
    }

    // Allocate device memory
    hipMalloc(&d_params, N * sizeof(float));
    hipMalloc(&d_grads, N * sizeof(float));
    hipMalloc(&d_m, N * sizeof(float));
    hipMalloc(&d_v, N * sizeof(float));

    // Copy data to device
    hipMemcpy(d_params, h_params, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_grads, h_grads, N * sizeof(float), hipMemcpyHostToDevice);

    // Initialize m and v to 0
    initKernel<<<1, N>>>(d_m, 0.0f);
    initKernel<<<1, N>>>(d_v, 0.0f);
    hipDeviceSynchronize();

    // Check for errors in memory allocation and initialization
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform multiple steps of Adam optimization
    for (int step = 1; step <= 100; ++step) {
        adamUpdateKernel<<<1, N, N * sizeof(float)>>>(d_params, d_grads, d_m, d_v, step);
        hipDeviceSynchronize();

        // Check for errors in kernel execution
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    hipMemcpy(h_params, d_params, N * sizeof(float), hipMemcpyDeviceToHost);

    // Output final parameters
    std::cout << "Final parameters:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_params[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_params);
    hipFree(d_grads);
    hipFree(d_m);
    hipFree(d_v);

    // Free host memory
    delete[] h_params;
    delete[] h_grads;

    return 0;
}