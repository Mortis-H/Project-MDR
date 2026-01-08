#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 2048
#define GENES_PER_INDIVIDUAL 128
#define BLOCK_SIZE 256

__global__ void initializePopulation(hiprandState *states, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            population[idx * GENES_PER_INDIVIDUAL + i] = hiprand_uniform(&localState);
        }
        states[idx] = localState;
    }
}

__global__ void geneticAlgorithmStep(float *population, float *fitness, float *newPopulation) {
    extern __shared__ float sharedPopulation[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sharedIdx = threadIdx.x;

    // Load individual into shared memory
    for (int i = 0; i < GENES_PER_INDIVIDUAL; i += blockDim.x) {
        int geneIdx = i + sharedIdx;
        if (geneIdx < GENES_PER_INDIVIDUAL) {
            sharedPopulation[geneIdx] = population[idx * GENES_PER_INDIVIDUAL + geneIdx];
        }
        __syncthreads();
    }

    // Perform genetic algorithm operations (e.g., crossover, mutation)
    // Here we just simulate a simple operation for demonstration
    if (sharedIdx < GENES_PER_INDIVIDUAL) {
        sharedPopulation[sharedIdx] *= 1.01f; // Example mutation
    }
    __syncthreads();

    // Write back to global memory
    for (int i = 0; i < GENES_PER_INDIVIDUAL; i += blockDim.x) {
        int geneIdx = i + sharedIdx;
        if (geneIdx < GENES_PER_INDIVIDUAL) {
            newPopulation[idx * GENES_PER_INDIVIDUAL + geneIdx] = sharedPopulation[geneIdx];
        }
    }
}

void checkCudaError(hipError_t error, const char *operation) {
    if (error != hipSuccess) {
        std::cerr << "CUDA error during " << operation << ": " << hipGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    hiprandState *d_states;
    float *d_population, *d_newPopulation, *d_fitness;

    // Allocate memory for population, new population, and fitness
    checkCudaError(hipMalloc(&d_population, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float)), "hipMalloc population");
    checkCudaError(hipMalloc(&d_newPopulation, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float)), "hipMalloc newPopulation");
    checkCudaError(hipMalloc(&d_fitness, POPULATION_SIZE * sizeof(float)), "hipMalloc fitness");
    checkCudaError(hipMalloc(&d_states, POPULATION_SIZE * sizeof(hiprandState)), "hipMalloc states");

    // Initialize curand states
    unsigned int seed = 1234;
    int numBlocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializePopulation<<<numBlocks, BLOCK_SIZE>>>(d_states, d_population);
    checkCudaError(hipGetLastError(), "initializePopulation kernel launch");

    // Perform genetic algorithm step
    geneticAlgorithmStep<<<numBlocks, BLOCK_SIZE, GENES_PER_INDIVIDUAL * sizeof(float)>>>(d_population, d_fitness, d_newPopulation);
    checkCudaError(hipGetLastError(), "geneticAlgorithmStep kernel launch");

    // Copy results back to host for verification (optional)
    float *h_population = new float[POPULATION_SIZE * GENES_PER_INDIVIDUAL];
    checkCudaError(hipMemcpy(h_population, d_newPopulation, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy");

    // Free allocated memory
    delete[] h_population;
    checkCudaError(hipFree(d_population), "hipFree population");
    checkCudaError(hipFree(d_newPopulation), "hipFree newPopulation");
    checkCudaError(hipFree(d_fitness), "hipFree fitness");
    checkCudaError(hipFree(d_states), "hipFree states");

    return 0;
}