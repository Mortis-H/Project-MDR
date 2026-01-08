// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 8192
#define CHROMOSOME_LENGTH 128
#define NUM_GENERATIONS 100

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &states[idx]);
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            population[idx * CHROMOSOME_LENGTH + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness (simple example: count number of 1s)
__global__ void evaluateFitness(int *population, int *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int fit = 0;
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            fit += population[idx * CHROMOSOME_LENGTH + i];
        }
        fitness[idx] = fit;
    }
}

// Kernel to perform selection, crossover, and mutation
__global__ void evolvePopulation(hiprandState *states, int *population, int *fitness) {
    extern __shared__ int sharedFitness[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        int parent1Idx = idx;
        int parent2Idx = hiprand(&localState) % POPULATION_SIZE;
        int crossoverPoint = hiprand(&localState) % CHROMOSOME_LENGTH;

        // Simple selection based on fitness
        for (int i = 0; i < blockDim.x; ++i) {
            int otherIdx = blockIdx.x * blockDim.x + i;
            if (otherIdx < POPULATION_SIZE && fitness[otherIdx] > fitness[parent1Idx]) {
                parent1Idx = otherIdx;
            }
        }

        // Load fitness into shared memory for faster access
        sharedFitness[threadIdx.x] = fitness[parent1Idx];
        __syncthreads();

        // Crossover
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            if (i < crossoverPoint) {
                population[idx * CHROMOSOME_LENGTH + i] = population[parent1Idx * CHROMOSOME_LENGTH + i];
            } else {
                population[idx * CHROMOSOME_LENGTH + i] = population[parent2Idx * CHROMOSOME_LENGTH + i];
            }
        }

        // Mutation
        if (hiprand(&localState) < 0.01f) {
            int mutationPoint = hiprand(&localState) % CHROMOSOME_LENGTH;
            population[idx * CHROMOSOME_LENGTH + mutationPoint] = 1 - population[idx * CHROMOSOME_LENGTH + mutationPoint];
        }

        states[idx] = localState;
    }
}

int main() {
    int *population, *fitness;
    hiprandState *states;
    size_t populationSizeBytes = POPULATION_SIZE * CHROMOSOME_LENGTH * sizeof(int);
    size_t fitnessSizeBytes = POPULATION_SIZE * sizeof(int);
    size_t statesSizeBytes = POPULATION_SIZE * sizeof(hiprandState);

    // Allocate Unified Memory
    hipError_t err = hipMallocManaged(&population, populationSizeBytes);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate Unified Memory for population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMallocManaged(&fitness, fitnessSizeBytes);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate Unified Memory for fitness: " << hipGetErrorString(err) << std::endl;
        hipFree(population);
        return -1;
    }
    err = hipMallocManaged(&states, statesSizeBytes);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate Unified Memory for states: " << hipGetErrorString(err) << std::endl;
        hipFree(population);
        hipFree(fitness);
        return -1;
    }

    // Set up execution configuration
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;

    // Initialize population
    initializePopulation<<<numBlocks, blockSize>>>(states, population);
    hipDeviceSynchronize();

    // Genetic algorithm loop
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize>>>(population, fitness);
        hipDeviceSynchronize();

        // Evolve population
        evolvePopulation<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(states, population, fitness);
        hipDeviceSynchronize();
    }

    // Free Unified Memory
    hipFree(population);
    hipFree(fitness);
    hipFree(states);

    return 0;
}