// particle_swarm_optimization.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 512
#define DIMENSIONS 6

// Kernel to initialize particles
__global__ void initParticles(float *positions, float *velocities, float *pBestPositions, float *pBestValues, hiprandState *states, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        for (int d = 0; d < DIMENSIONS; ++d) {
            positions[idx * DIMENSIONS + d] = hiprand_uniform(&localState) * (maxVal - minVal) + minVal;
            velocities[idx * DIMENSIONS + d] = 0.0f;
            pBestPositions[idx * DIMENSIONS + d] = positions[idx * DIMENSIONS + d];
        }
        pBestValues[idx] = 1e30f; // Initialize with a large value
        states[idx] = localState;
    }
}

// Kernel to perform the particle update
__global__ void updateParticles(float *positions, float *velocities, float *pBestPositions, float *pBestValues, float *gBestPosition, float w, float c1, float c2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &localState);
        for (int d = 0; d < DIMENSIONS; ++d) {
            float r1 = hiprand_uniform(&localState);
            float r2 = hiprand_uniform(&localState);
            velocities[idx * DIMENSIONS + d] = w * velocities[idx * DIMENSIONS + d] +
                                              c1 * r1 * (pBestPositions[idx * DIMENSIONS + d] - positions[idx * DIMENSIONS + d]) +
                                              c2 * r2 * (gBestPosition[d] - positions[idx * DIMENSIONS + d]);
            positions[idx * DIMENSIONS + d] += velocities[idx * DIMENSIONS + d];
        }
        __syncthreads(); // Ensure all threads in the block have updated their positions

        // Evaluate fitness (simple sphere function)
        float fitness = 0.0f;
        for (int d = 0; d < DIMENSIONS; ++d) {
            fitness += positions[idx * DIMENSIONS + d] * positions[idx * DIMENSIONS + d];
        }

        // Update personal best
        if (fitness < pBestValues[idx]) {
            pBestValues[idx] = fitness;
            for (int d = 0; d < DIMENSIONS; ++d) {
                pBestPositions[idx * DIMENSIONS + d] = positions[idx * DIMENSIONS + d];
            }
        }
    }
}

int main() {
    float *d_positions, *d_velocities, *d_pBestPositions, *d_pBestValues, *d_gBestPosition;
    hiprandState *d_states;

    // Allocate memory on the device
    hipMalloc((void**)&d_positions, NUM_PARTICLES * DIMENSIONS * sizeof(float));
    hipMalloc((void**)&d_velocities, NUM_PARTICLES * DIMENSIONS * sizeof(float));
    hipMalloc((void**)&d_pBestPositions, NUM_PARTICLES * DIMENSIONS * sizeof(float));
    hipMalloc((void**)&d_pBestValues, NUM_PARTICLES * sizeof(float));
    hipMalloc((void**)&d_gBestPosition, DIMENSIONS * sizeof(float));
    hipMalloc((void**)&d_states, NUM_PARTICLES * sizeof(hiprandState));

    // Initialize particles
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
    initParticles<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_velocities, d_pBestPositions, d_pBestValues, d_states, -5.0f, 5.0f);

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initParticles kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Main loop for particle swarm optimization
    float w = 0.729; // Inertia weight
    float c1 = 1.49445; // Cognitive coefficient
    float c2 = 1.49445; // Social coefficient
    int iterations = 1000;

    for (int iter = 0; iter < iterations; ++iter) {
        // Update particles
        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_velocities, d_pBestPositions, d_pBestValues, d_gBestPosition, w, c1, c2);

        // Check for errors in kernel launch
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching updateParticles kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Synchronize device to ensure all updates are complete
        hipDeviceSynchronize();

        // Find global best position and value
        float h_gBestValue = 1e30f;
        float h_gBestPosition[DIMENSIONS];
        float *h_pBestValues = new float[NUM_PARTICLES];
        hipMemcpy(h_pBestValues, d_pBestValues, NUM_PARTICLES * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(d_positions, d_pBestPositions, NUM_PARTICLES * DIMENSIONS * sizeof(float), hipMemcpyDeviceToDevice);

        for (int i = 0; i < NUM_PARTICLES; ++i) {
            if (h_pBestValues[i] < h_gBestValue) {
                h_gBestValue = h_pBestValues[i];
                hipMemcpy(h_gBestPosition, d_pBestPositions + i * DIMENSIONS, DIMENSIONS * sizeof(float), hipMemcpyDeviceToHost);
            }
        }

        hipMemcpy(d_gBestPosition, h_gBestPosition, DIMENSIONS * sizeof(float), hipMemcpyHostToDevice);

        delete[] h_pBestValues;
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_pBestPositions);
    hipFree(d_pBestValues);
    hipFree(d_gBestPosition);
    hipFree(d_states);

    return 0;
}