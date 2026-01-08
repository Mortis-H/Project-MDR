#include <iostream>
#include <hip/hip_runtime.h>


#define N 16384
#define BLOCK_SIZE 256

__device__ float f[N][9]; // Distribution functions
__device__ float f_eq[N][9]; // Equilibrium distribution functions
__device__ float rho[N]; // Density
__device__ float u[N][2]; // Velocity vector
__device__ float omega = 1.0f / 0.6f; // Relaxation parameter
__device__ float w[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f}; // Weights
__device__ int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Direction vectors x
__device__ int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1}; // Direction vectors y

__global__ void computeEquilibrium(float *rho, float *u, float *f_eq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float ux = u[idx * 2];
        float uy = u[idx * 2 + 1];
        float cu[9];
        float cu2[9];
        float uu = ux * ux + uy * uy;
        for (int i = 0; i < 9; ++i) {
            cu[i] = cx[i] * ux + cy[i] * uy;
            cu2[i] = cu[i] * cu[i];
            f_eq[idx * 9 + i] = w[i] * rho[idx] * (1.0f + 3.0f * cu[i] + 4.5f * cu2[i] - 1.5f * uu);
        }
    }
}

__global__ void collideAndStream(float *f, float *f_eq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 9; ++i) {
            f[idx * 9 + i] = f[idx * 9 + i] - omega * (f[idx * 9 + i] - f_eq[idx * 9 + i]);
            int nx = (idx + cx[i] + N) % N;
            f[nx * 9 + (8 - i)] = f[idx * 9 + i];
        }
    }
}

__global__ void computeDensityAndVelocity(float *f, float *rho, float *u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        rho[idx] = 0.0f;
        u[idx * 2] = 0.0f;
        u[idx * 2 + 1] = 0.0f;
        for (int i = 0; i < 9; ++i) {
            rho[idx] += f[idx * 9 + i];
            u[idx * 2] += f[idx * 9 + i] * cx[i];
            u[idx * 2 + 1] += f[idx * 9 + i] * cy[i];
        }
        u[idx * 2] /= rho[idx];
        u[idx * 2 + 1] /= rho[idx];
    }
}

__global__ void initialize(float *f, float *rho, float *u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        rho[idx] = 1.0f;
        u[idx * 2] = 0.0f;
        u[idx * 2 + 1] = 0.0f;
        for (int i = 0; i < 9; ++i) {
            f[idx * 9 + i] = w[i] * rho[idx];
        }
    }
}

int main() {
    float *d_f, *d_f_eq, *d_rho, *d_u;

    hipError_t err;

    // Allocate memory on the device
    err = hipMalloc((void**)&d_f, N * 9 * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for f: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void**)&d_f_eq, N * 9 * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for f_eq: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void**)&d_rho, N * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for rho: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void**)&d_u, N * 2 * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for u: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Initialize the simulation
    initialize<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_f, d_rho, d_u);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error synchronizing after initialization: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Perform a few iterations of the lattice Boltzmann method
    for (int iter = 0; iter < 100; ++iter) {
        computeEquilibrium<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_rho, d_u, d_f_eq);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error computing equilibrium: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        collideAndStream<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_f, d_f_eq);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error colliding and streaming: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        computeDensityAndVelocity<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_f, d_rho, d_u);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error computing density and velocity: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Error synchronizing after iteration " << iter << ": " << hipGetErrorString(err) << std::endl;
            return 1;
        }
    }

    // Free device memory
    hipFree(d_f);
    hipFree(d_f_eq);
    hipFree(d_rho);
    hipFree(d_u);

    return 0;
}