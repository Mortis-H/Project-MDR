#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 64
#define DT 0.01f
#define CUTOFF 1.0f
#define EPSILON 1.0f
#define SIGMA 1.0f
#define BLOCK_SIZE 32

// Kernel to calculate forces using Lennard-Jones potential
__global__ void calculateForces(float4* pos, float4* force, float4* posPrev) {
    extern __shared__ float4 sharedPos[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load positions into shared memory
    if (idx < N) {
        sharedPos[tid] = pos[idx];
    }
    __syncthreads();

    // Initialize force to zero
    float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Calculate forces using Lennard-Jones potential
    for (int i = 0; i < N; i += blockDim.x) {
        float4 r = sharedPos[(tid + i) % N];
        float dx = r.x - pos[idx].x;
        float dy = r.y - pos[idx].y;
        float dz = r.z - pos[idx].z;
        float r2 = dx * dx + dy * dy + dz * dz;

        if (r2 < CUTOFF * CUTOFF && r2 > 0.0f) {
            float r6 = r2 * r2 * r2;
            float fmag = 24.0f * EPSILON * (2.0f / r6 / r6 - 1.0f / r6) / r2;
            f.x += fmag * dx;
            f.y += fmag * dy;
            f.z += fmag * dz;
        }
    }
    __syncthreads();

    // Store forces
    if (idx < N) {
        force[idx] = f;
    }
}

// Kernel to update positions using Verlet integration
__global__ void updatePositions(float4* pos, float4* force, float4* posPrev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float4 p = pos[idx];
        float4 f = force[idx];
        float4 pPrev = posPrev[idx];

        // Verlet integration
        float4 pNew = make_float4(
            2.0f * p.x - pPrev.x + f.x * DT * DT,
            2.0f * p.y - pPrev.y + f.y * DT * DT,
            2.0f * p.z - pPrev.z + f.z * DT * DT,
            0.0f
        );

        // Update positions and previous positions
        posPrev[idx] = p;
        pos[idx] = pNew;
    }
}

int main() {
    // Initialize random number generator
    hiprandState* devStates;
    hipMalloc((void**)&devStates, N * sizeof(hiprandState));
    hiprandState* hostStates = (hiprandState*)malloc(N * sizeof(hiprandState));

    // Initialize positions and previous positions
    float4* d_pos, *d_force, *d_posPrev;
    hipMalloc((void**)&d_pos, N * sizeof(float4));
    hipMalloc((void**)&d_force, N * sizeof(float4));
    hipMalloc((void**)&d_posPrev, N * sizeof(float4));

    // Initialize positions randomly
    float4* h_pos = (float4*)malloc(N * sizeof(float4));
    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, 0.0f);
    }
    hipMemcpy(d_pos, h_pos, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(d_posPrev, h_pos, N * sizeof(float4), hipMemcpyHostToDevice);

    // Simulation loop
    int steps = 100;
    for (int step = 0; step < steps; step++) {
        calculateForces<<<N / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float4)>>>(d_pos, d_force, d_posPrev);
        hipDeviceSynchronize();
        updatePositions<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_pos, d_force, d_posPrev);
        hipDeviceSynchronize();

        // Error checking
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free memory
    hipFree(d_pos);
    hipFree(d_force);
    hipFree(d_posPrev);
    free(h_pos);
    free(hostStates);
    hipFree(devStates);

    return 0;
}