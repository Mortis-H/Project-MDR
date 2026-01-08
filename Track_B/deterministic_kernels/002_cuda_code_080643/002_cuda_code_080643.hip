#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#define NUM_PARTICLES 8192
#define DIMENSIONS 5
#define NUM_GPUS 2

// CUDA error check macro
#define cudaCheckErrors(msg) \
    do { \
        hipError_t __err = hipGetLastError(); \
        if (__err != hipSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, hipGetErrorString(__err), \
                    __FILE__, __LINE__); \
            exit(-1); \
        } \
    } while (0)

__global__ void initializeParticles(float *positions, float *velocities, float *pBestPositions, float *pBestValues, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        for (int d = 0; d < DIMENSIONS; ++d) {
            positions[idx * DIMENSIONS + d] = hiprand_uniform(&localState) * 100.0f - 50.0f;
            velocities[idx * DIMENSIONS + d] = hiprand_uniform(&localState) * 2.0f - 1.0f;
            pBestPositions[idx * DIMENSIONS + d] = positions[idx * DIMENSIONS + d];
        }
        pBestValues[idx] = 1e30f;
        states[idx] = localState;
    }
}

__device__ float fitnessFunction(float *position) {
    float sum = 0.0f;
    for (int d = 0; d < DIMENSIONS; ++d) {
        sum += position[d] * position[d];
    }
    return sum;
}

__global__ void updateParticles(float *positions, float *velocities, float *pBestPositions, float *pBestValues, float *gBestPosition, float w, float c1, float c2, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        float newVelocity[DIMENSIONS];
        float newPosition[DIMENSIONS];
        float currentFitness;

        for (int d = 0; d < DIMENSIONS; ++d) {
            float r1 = hiprand_uniform(&localState);
            float r2 = hiprand_uniform(&localState);
            newVelocity[d] = w * velocities[idx * DIMENSIONS + d] +
                             c1 * r1 * (pBestPositions[idx * DIMENSIONS + d] - positions[idx * DIMENSIONS + d]) +
                             c2 * r2 * (gBestPosition[d] - positions[idx * DIMENSIONS + d]);
            newPosition[d] = positions[idx * DIMENSIONS + d] + newVelocity[d];
        }

        currentFitness = fitnessFunction(newPosition);

        if (currentFitness < pBestValues[idx]) {
            pBestValues[idx] = currentFitness;
            for (int d = 0; d < DIMENSIONS; ++d) {
                pBestPositions[idx * DIMENSIONS + d] = newPosition[d];
            }
        }

        for (int d = 0; d < DIMENSIONS; ++d) {
            positions[idx * DIMENSIONS + d] = newPosition[d];
            velocities[idx * DIMENSIONS + d] = newVelocity[d];
        }

        states[idx] = localState;
    }
}

int main() {
    float *h_positions, *h_velocities, *h_pBestPositions, *h_pBestValues;
    float *d_positions, *d_velocities, *d_pBestPositions, *d_pBestValues, *d_gBestPosition;
    hiprandState *d_states;
    int numThreads = 256;
    int numBlocks = (NUM_PARTICLES + numThreads - 1) / numThreads;
    float w = 0.7f, c1 = 1.5f, c2 = 1.5f;
    float gBestValue = 1e30f;
    float h_gBestPosition[DIMENSIONS] = {0};

    // Allocate host memory
    h_positions = new float[NUM_PARTICLES * DIMENSIONS];
    h_velocities = new float[NUM_PARTICLES * DIMENSIONS];
    h_pBestPositions = new float[NUM_PARTICLES * DIMENSIONS];
    h_pBestValues = new float[NUM_PARTICLES];

    // Allocate device memory
    hipMalloc(&d_positions, NUM_PARTICLES * DIMENSIONS * sizeof(float));
    hipMalloc(&d_velocities, NUM_PARTICLES * DIMENSIONS * sizeof(float));
    hipMalloc(&d_pBestPositions, NUM_PARTICLES * DIMENSIONS * sizeof(float));
    hipMalloc(&d_pBestValues, NUM_PARTICLES * sizeof(float));
    hipMalloc(&d_gBestPosition, DIMENSIONS * sizeof(float));
    hipMalloc(&d_states, NUM_PARTICLES * sizeof(hiprandState));

    // Initialize particles
    initializeParticles<<<numBlocks, numThreads>>>(d_positions, d_velocities, d_pBestPositions, d_pBestValues, d_states, 1234);
    cudaCheckErrors("initializeParticles kernel launch failed");

    // Copy global best position to device
    hipMemcpy(d_gBestPosition, h_gBestPosition, DIMENSIONS * sizeof(float), hipMemcpyHostToDevice);

    // Perform PSO iterations
    for (int iter = 0; iter < 1000; ++iter) {
        updateParticles<<<numBlocks, numThreads>>>(d_positions, d_velocities, d_pBestPositions, d_pBestValues, d_gBestPosition, w, c1, c2, d_states);
        cudaCheckErrors("updateParticles kernel launch failed");

        // Find global best
        thrust::device_ptr<float> d_pBestValues_ptr(d_pBestValues);
        int minIndex = thrust::min_element(d_pBestValues_ptr, d_pBestValues_ptr + NUM_PARTICLES) - d_pBestValues_ptr;
        hipMemcpy(&gBestValue, d_pBestValues + minIndex, sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(h_gBestPosition, d_pBestPositions + minIndex * DIMENSIONS, DIMENSIONS * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(d_gBestPosition, h_gBestPosition, DIMENSIONS * sizeof(float), hipMemcpyHostToDevice);
    }

    // Free device memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_pBestPositions);
    hipFree(d_pBestValues);
    hipFree(d_gBestPosition);
    hipFree(d_states);

    // Free host memory
    delete[] h_positions;
    delete[] h_velocities;
    delete[] h_pBestPositions;
    delete[] h_pBestValues;

    std::cout << "Global Best Value: " << gBestValue << std::endl;
    std::cout << "Global Best Position: ";
    for (int d = 0; d < DIMENSIONS; ++d) {
        std::cout << h_gBestPosition[d] << " ";
    }
    std::cout << std::endl;

    return 0;
}