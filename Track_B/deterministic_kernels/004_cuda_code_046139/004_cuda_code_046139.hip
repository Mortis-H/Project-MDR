#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Constants for the simulation
#define NX 512
#define NY 512
#define NUM_DIRECTIONS 9
#define OMEGA 0.65f // Relaxation parameter
#define MAX_ITERATIONS 1000

// Lattice velocities and weights
__constant__ int cx[NUM_DIRECTIONS] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int cy[NUM_DIRECTIONS] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant__ float w[NUM_DIRECTIONS] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

// Initialize the lattice distribution functions
__global__ void initialize(float *f, float *rho, float *ux, float *uy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * NX + idx;

    if (idx < NX && idy < NY) {
        rho[index] = 1.0f; // Initial density
        ux[index] = 0.0f;  // Initial x-velocity
        uy[index] = 0.0f;  // Initial y-velocity

        for (int i = 0; i < NUM_DIRECTIONS; ++i) {
            f[index * NUM_DIRECTIONS + i] = w[i] * rho[index];
        }
    }
}

// Streaming step of the LBM
__global__ void stream(float *f, float *f_new) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * NX + idx;

    if (idx < NX && idy < NY) {
        for (int i = 0; i < NUM_DIRECTIONS; ++i) {
            int nx = idx + cx[i];
            int ny = idy + cy[i];

            // Periodic boundary conditions
            if (nx < 0) nx = NX - 1;
            if (nx >= NX) nx = 0;
            if (ny < 0) ny = NY - 1;
            if (ny >= NY) ny = 0;

            int nindex = ny * NX + nx;
            f_new[nindex * NUM_DIRECTIONS + i] = f[index * NUM_DIRECTIONS + i];
        }
    }
}

// Collision step of the LBM
__global__ void collide(float *f_new, float *rho, float *ux, float *uy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * NX + idx;

    if (idx < NX && idy < NY) {
        float uxsq = ux[index] * ux[index];
        float uysq = uy[index] * uy[index];
        float usq = uxsq + uysq;
        float uxuy = ux[index] * uy[index];

        for (int i = 0; i < NUM_DIRECTIONS; ++i) {
            float cu = cx[i] * ux[index] + cy[i] * uy[index];
            float feq = w[i] * rho[index] * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
            f_new[index * NUM_DIRECTIONS + i] = f_new[index * NUM_DIRECTIONS + i] - OMEGA * (f_new[index * NUM_DIRECTIONS + i] - feq);
        }
    }
}

// Compute density and velocity
__global__ void compute_rho_ux_uy(float *f_new, float *rho, float *ux, float *uy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * NX + idx;

    if (idx < NX && idy < NY) {
        rho[index] = 0.0f;
        ux[index] = 0.0f;
        uy[index] = 0.0f;

        for (int i = 0; i < NUM_DIRECTIONS; ++i) {
            rho[index] += f_new[index * NUM_DIRECTIONS + i];
            ux[index] += f_new[index * NUM_DIRECTIONS + i] * cx[i];
            uy[index] += f_new[index * NUM_DIRECTIONS + i] * cy[i];
        }

        ux[index] /= rho[index];
        uy[index] /= rho[index];
    }
}

int main() {
    // Allocate unified memory
    float *f, *f_new, *rho, *ux, *uy;
    hipMallocManaged(&f, NX * NY * NUM_DIRECTIONS * sizeof(float));
    hipMallocManaged(&f_new, NX * NY * NUM_DIRECTIONS * sizeof(float));
    hipMallocManaged(&rho, NX * NY * sizeof(float));
    hipMallocManaged(&ux, NX * NY * sizeof(float));
    hipMallocManaged(&uy, NX * NY * sizeof(float));

    // Initialize the lattice
    dim3 blockSize(16, 16);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);
    initialize<<<gridSize, blockSize>>>(f, rho, ux, uy);

    // Perform the simulation
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        stream<<<gridSize, blockSize>>>(f, f_new);
        hipDeviceSynchronize();

        collide<<<gridSize, blockSize>>>(f_new, rho, ux, uy);
        hipDeviceSynchronize();

        compute_rho_ux_uy<<<gridSize, blockSize>>>(f_new, rho, ux, uy);
        hipDeviceSynchronize();

        // Swap f and f_new for the next iteration
        float *temp = f;
        f = f_new;
        f_new = temp;
    }

    // Free unified memory
    hipFree(f);
    hipFree(f_new);
    hipFree(rho);
    hipFree(ux);
    hipFree(uy);

    return 0;
}