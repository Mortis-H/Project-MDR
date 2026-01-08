// genetic_algorithm.cu
// A basic CUDA implementation of a genetic algorithm with dynamic parallelism
// Target architecture: Ampere

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 256
#define GENES_PER_INDIVIDUAL 16
#define MAX_GENERATIONS 10

// Kernel to initialize the population with random genes
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            population[idx * GENES_PER_INDIVIDUAL + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness of each individual
__global__ void evaluateFitness(int *population, int *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int fit = 0;
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            fit += population[idx * GENES_PER_INDIVIDUAL + i];
        }
        fitness[idx] = fit;
    }
}

// Kernel to perform selection and crossover
__global__ void selectionAndCrossover(int *population, int *fitness, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        int parent1 = hiprand(&localState) % POPULATION_SIZE;
        int parent2 = hiprand(&localState) % POPULATION_SIZE;
        int crossoverPoint = hiprand(&localState) % GENES_PER_INDIVIDUAL;

        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            if (i < crossoverPoint) {
                population[idx * GENES_PER_INDIVIDUAL + i] = population[parent1 * GENES_PER_INDIVIDUAL + i];
            } else {
                population[idx * GENES_PER_INDIVIDUAL + i] = population[parent2 * GENES_PER_INDIVIDUAL + i];
            }
        }
        states[idx] = localState;
    }
}

// Kernel to perform mutation
__global__ void mutatePopulation(int *population, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            if (hiprand(&localState) % 100 < 5) { // 5% mutation rate
                population[idx * GENES_PER_INDIVIDUAL + i] = 1 - population[idx * GENES_PER_INDIVIDUAL + i];
            }
        }
        states[idx] = localState;
    }
}

int main() {
    int *d_population, *d_fitness;
    hiprandState *d_states;

    // Allocate memory on the device
    hipMalloc((void**)&d_population, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(int));
    hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(int));
    hipMalloc((void**)&d_states, POPULATION_SIZE * sizeof(hiprandState));

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    hipDeviceSynchronize();

    // Main genetic algorithm loop
    for (int generation = 0; generation < MAX_GENERATIONS; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        hipDeviceSynchronize();

        // Selection and crossover
        selectionAndCrossover<<<numBlocks, blockSize>>>(d_population, d_fitness, d_states);
        hipDeviceSynchronize();

        // Mutate population
        mutatePopulation<<<numBlocks, blockSize>>>(d_population, d_states);
        hipDeviceSynchronize();
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);

    return 0;
}