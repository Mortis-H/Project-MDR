// genetic_algorithm.cu
// A simple genetic algorithm implemented in CUDA using the persistent threads design pattern.
// Optimized for register usage and targeted for Turing architecture.

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 256
#define GENES 32
#define MUTATION_RATE 0.01f

// Kernel function to initialize the population with random genes
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENES; i++) {
            population[idx * GENES + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel function to evaluate fitness of each individual
__global__ void evaluateFitness(int *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int sum = 0;
        for (int i = 0; i < GENES; i++) {
            sum += population[idx * GENES + i];
        }
        fitness[idx] = (float)sum / GENES;
    }
}

// Kernel function to perform crossover and mutation
__global__ void crossoverAndMutate(hiprandState *states, int *population, float *fitness) {
    extern __shared__ int shared_population[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        int parent1 = idx;
        int parent2 = (idx + 1) % POPULATION_SIZE;

        // Select parents based on fitness
        for (int i = 0; i < POPULATION_SIZE; i++) {
            if (hiprand_uniform(&localState) < fitness[i]) {
                parent1 = i;
            }
            if (hiprand_uniform(&localState) < fitness[i]) {
                parent2 = i;
            }
        }

        // Crossover
        int crossoverPoint = hiprand(&localState) % GENES;
        for (int i = 0; i < GENES; i++) {
            shared_population[threadIdx.x * GENES + i] = (i < crossoverPoint) ?
                                                      population[parent1 * GENES + i] :
                                                      population[parent2 * GENES + i];
        }

        // Mutation
        for (int i = 0; i < GENES; i++) {
            if (hiprand_uniform(&localState) < MUTATION_RATE) {
                shared_population[threadIdx.x * GENES + i] = 1 - shared_population[threadIdx.x * GENES + i];
            }
        }

        // Write back to global memory
        for (int i = 0; i < GENES; i++) {
            population[idx * GENES + i] = shared_population[threadIdx.x * GENES + i];
        }
        states[idx] = localState;
    }
}

int main() {
    int *d_population, *h_population;
    float *d_fitness;
    hiprandState *d_states;
    size_t populationSizeBytes = POPULATION_SIZE * GENES * sizeof(int);
    size_t fitnessSizeBytes = POPULATION_SIZE * sizeof(float);
    size_t statesSizeBytes = POPULATION_SIZE * sizeof(hiprandState);

    // Allocate memory
    hipMalloc((void **)&d_population, populationSizeBytes);
    hipMalloc((void **)&d_fitness, fitnessSizeBytes);
    hipMalloc((void **)&d_states, statesSizeBytes);
    h_population = (int *)malloc(populationSizeBytes);

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    hipDeviceSynchronize();

    // Evaluate fitness
    evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
    hipDeviceSynchronize();

    // Crossover and mutate
    size_t sharedMemSize = blockSize * GENES * sizeof(int);
    crossoverAndMutate<<<numBlocks, blockSize, sharedMemSize>>>(d_states, d_population, d_fitness);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_population, d_population, populationSizeBytes, hipMemcpyDeviceToHost);

    // Free memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);
    free(h_population);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Genetic algorithm completed successfully." << std::endl;
    return 0;
}