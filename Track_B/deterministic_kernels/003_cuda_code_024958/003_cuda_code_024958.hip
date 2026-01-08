// genetic_algorithm.cu
// A CUDA implementation of a simple genetic algorithm with multi-GPU scaling focus.
// Target architecture: Volta

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 512
#define GENES 1024
#define GENERATIONS 1000

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < GENES; ++i) {
            population[idx * GENES + i] = hiprand_uniform(&states[idx]);
        }
    }
}

// Kernel to calculate fitness of each individual
__global__ void calculateFitness(float *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < GENES; ++i) {
            sum += population[idx * GENES + i];
        }
        fitness[idx] = sum / GENES; // Simple fitness function
    }
}

// Kernel to perform selection, crossover, and mutation
__global__ void evolvePopulation(float *population, float *fitness, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Selection: Tournament selection
        int parent1 = idx;
        int parent2 = (idx + 1) % POPULATION_SIZE;
        if (fitness[parent2] > fitness[parent1]) {
            parent1 = parent2;
        }

        // Crossover: Single-point crossover
        int crossoverPoint = hiprand(&states[idx]) % GENES;
        for (int i = 0; i < crossoverPoint; ++i) {
            population[idx * GENES + i] = population[parent1 * GENES + i];
        }
        for (int i = crossoverPoint; i < GENES; ++i) {
            population[idx * GENES + i] = population[parent2 * GENES + i];
        }

        // Mutation: Random mutation with low probability
        for (int i = 0; i < GENES; ++i) {
            if (hiprand_uniform(&states[idx]) < 0.01f) {
                population[idx * GENES + i] = hiprand_uniform(&states[idx]);
            }
        }
    }
}

// Utility function to check for CUDA errors
void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *d_population, *d_fitness;
    hiprandState *d_states;

    // Allocate memory for population, fitness, and random states
    checkCudaError(hipMalloc((void **)&d_population, POPULATION_SIZE * GENES * sizeof(float)), "hipMalloc d_population");
    checkCudaError(hipMalloc((void **)&d_fitness, POPULATION_SIZE * sizeof(float)), "hipMalloc d_fitness");
    checkCudaError(hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState)), "hipMalloc d_states");

    // Initialize random states
    unsigned int seed = 1234;
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    checkCudaError(hipGetLastError(), "initializePopulation kernel launch");

    // Genetic algorithm loop
    for (int generation = 0; generation < GENERATIONS; ++generation) {
        // Calculate fitness
        calculateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        checkCudaError(hipGetLastError(), "calculateFitness kernel launch");

        // Evolve population
        evolvePopulation<<<numBlocks, blockSize>>>(d_population, d_fitness, d_states);
        checkCudaError(hipGetLastError(), "evolvePopulation kernel launch");
    }

    // Copy fitness results back to host
    float *h_fitness = new float[POPULATION_SIZE];
    checkCudaError(hipMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy d_fitness to h_fitness");

    // Find the best fitness
    float bestFitness = 0.0f;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (h_fitness[i] > bestFitness) {
            bestFitness = h_fitness[i];
        }
    }
    std::cout << "Best fitness: " << bestFitness << std::endl;

    // Clean up
    delete[] h_fitness;
    checkCudaError(hipFree(d_population), "hipFree d_population");
    checkCudaError(hipFree(d_fitness), "hipFree d_fitness");
    checkCudaError(hipFree(d_states), "hipFree d_states");

    return 0;
}