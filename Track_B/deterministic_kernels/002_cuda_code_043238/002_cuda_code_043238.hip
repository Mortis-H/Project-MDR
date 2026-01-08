// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 4096
#define GENES_PER_INDIVIDUAL 32
#define BLOCK_SIZE 256

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            population[idx * GENES_PER_INDIVIDUAL + i] = hiprand_uniform(&states[idx]);
        }
    }
}

// Kernel to perform genetic algorithm operations
__global__ void geneticAlgorithmKernel(float *population, float *fitness) {
    extern __shared__ float sharedMem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int baseIndex = idx * GENES_PER_INDIVIDUAL;

    // Load individual into shared memory to avoid bank conflicts
    for (int i = threadIdx.x; i < GENES_PER_INDIVIDUAL; i += blockDim.x) {
        sharedMem[i] = population[baseIndex + i];
    }
    __syncthreads();

    // Example operation: simple fitness function
    float localFitness = 0.0f;
    for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
        localFitness += sharedMem[i] * sharedMem[i];
    }

    // Store fitness result
    if (idx < POPULATION_SIZE) {
        fitness[idx] = localFitness;
    }
}

int main() {
    // Allocate memory for population and fitness on host
    float *h_population = new float[POPULATION_SIZE * GENES_PER_INDIVIDUAL];
    float *h_fitness = new float[POPULATION_SIZE];

    // Allocate memory for population and fitness on device
    float *d_population, *d_fitness;
    hipError_t err = hipMalloc((void**)&d_population, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device population memory" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device fitness memory" << std::endl;
        hipFree(d_population);
        return -1;
    }

    // Allocate and initialize CURAND states
    hiprandState *d_states;
    err = hipMalloc((void**)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate CURAND states memory" << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        return -1;
    }

    // Initialize population
    int numBlocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializePopulation<<<numBlocks, BLOCK_SIZE>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        hipFree(d_states);
        return -1;
    }

    // Perform genetic algorithm operations
    geneticAlgorithmKernel<<<numBlocks, BLOCK_SIZE, GENES_PER_INDIVIDUAL * sizeof(float)>>>(d_population, d_fitness);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        hipFree(d_states);
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy fitness results to host" << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        hipFree(d_states);
        return -1;
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);

    // Output the fitness of the first individual as a demonstration
    std::cout << "Fitness of the first individual: " << h_fitness[0] << std::endl;

    // Clean up host memory
    delete[] h_population;
    delete[] h_fitness;

    return 0;
}