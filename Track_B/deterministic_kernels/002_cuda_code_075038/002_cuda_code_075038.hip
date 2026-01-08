#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 16
#define N 4096

__global__ void initializeVelocityField(float* u, float* v, float* w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < N && idy < N) {
        int index = idx + idy * N;
        u[index] = 0.1f * sinf(2.0f * M_PI * idx / (float)N) * cosf(2.0f * M_PI * idy / (float)N);
        v[index] = -0.1f * cosf(2.0f * M_PI * idx / (float)N) * sinf(2.0f * M_PI * idy / (float)N);
        w[index] = 0.0f;
    }
}

__global__ void applyNavierStokes(float* u, float* v, float* w, float* u_new, float* v_new, float* w_new, float dt, float nu, float dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx > 0 && idx < N-1 && idy > 0 && idy < N-1) {
        int index = idx + idy * N;
        int left = index - 1;
        int right = index + 1;
        int up = index + N;
        int down = index - N;

        // Compute diffusion
        float laplacian_u = (u[right] + u[left] + u[up] + u[down] - 4.0f * u[index]) / (dx * dx);
        float laplacian_v = (v[right] + v[left] + v[up] + v[down] - 4.0f * v[index]) / (dx * dx);
        float laplacian_w = (w[right] + w[left] + w[up] + w[down] - 4.0f * w[index]) / (dx * dx);

        // Update velocities with diffusion
        u_new[index] = u[index] + dt * nu * laplacian_u;
        v_new[index] = v[index] + dt * nu * laplacian_v;
        w_new[index] = w[index] + dt * nu * laplacian_w;
    }
}

int main() {
    float* d_u, *d_v, *d_w, *d_u_new, *d_v_new, *d_w_new;
    size_t size = N * N * sizeof(float);
    float dt = 0.01f;
    float nu = 0.01f;
    float dx = 1.0f / (float)N;

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_u, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_v, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_w, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_u_new, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_v_new, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_w_new, size);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize velocity field
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    initializeVelocityField<<<dimGrid, dimBlock>>>(d_u, d_v, d_w);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA kernel error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Apply Navier-Stokes equations
    applyNavierStokes<<<dimGrid, dimBlock>>>(d_u, d_v, d_w, d_u_new, d_v_new, d_w_new, dt, nu, dx);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA kernel error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_u);
    hipFree(d_v);
    hipFree(d_w);
    hipFree(d_u_new);
    hipFree(d_v_new);
    hipFree(d_w_new);

    return 0;
}