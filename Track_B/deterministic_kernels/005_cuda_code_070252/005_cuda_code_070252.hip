// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 8192
#define GENES_PER_INDIVIDUAL 32
#define NUM_GENERATIONS 100

// Kernel to initialize random genes for each individual
__global__ void initializePopulation(hiprandState *state, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &state[idx]);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            population[idx * GENES_PER_INDIVIDUAL + i] = hiprand_uniform(&state[idx]);
        }
    }
}

// Kernel to evaluate fitness (simple example: sum of genes)
__global__ void evaluateFitness(float *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            sum += population[idx * GENES_PER_INDIVIDUAL + i];
        }
        fitness[idx] = sum;
    }
}

// Kernel to perform selection (simple example: tournament selection)
__global__ void selection(float *fitness, int *parents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int tournamentSize = 4;
        int winner = idx;
        for (int i = 1; i < tournamentSize; ++i) {
            int competitor = (idx + i) % POPULATION_SIZE;
            if (fitness[competitor] > fitness[winner]) {
                winner = competitor;
            }
        }
        parents[idx] = winner;
    }
}

// Kernel to perform crossover (simple example: single-point crossover)
__global__ void crossover(float *population, int *parents, float *newPopulation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = parents[idx];
        int parent2 = parents[(idx + 1) % POPULATION_SIZE];
        int crossoverPoint = GENES_PER_INDIVIDUAL / 2;
        for (int i = 0; i < crossoverPoint; ++i) {
            newPopulation[idx * GENES_PER_INDIVIDUAL + i] = population[parent1 * GENES_PER_INDIVIDUAL + i];
        }
        for (int i = crossoverPoint; i < GENES_PER_INDIVIDUAL; ++i) {
            newPopulation[idx * GENES_PER_INDIVIDUAL + i] = population[parent2 * GENES_PER_INDIVIDUAL + i];
        }
    }
}

// Kernel to perform mutation (simple example: flip a gene with a small probability)
__global__ void mutation(float *population, hiprandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            if (hiprand_uniform(&state[idx]) < 0.01f) {
                population[idx * GENES_PER_INDIVIDUAL + i] = 1.0f - population[idx * GENES_PER_INDIVIDUAL + i];
            }
        }
    }
}

int main() {
    float *d_population, *d_newPopulation, *d_fitness;
    int *d_parents;
    hiprandState *d_state;
    size_t populationSizeBytes = POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float);
    size_t fitnessSizeBytes = POPULATION_SIZE * sizeof(float);
    size_t parentsSizeBytes = POPULATION_SIZE * sizeof(int);
    size_t stateSizeBytes = POPULATION_SIZE * sizeof(hiprandState);

    // Allocate device memory
    hipError_t err = hipMalloc(&d_population, populationSizeBytes);
    if (err != hipSuccess) { std::cerr << "Error allocating d_population: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_newPopulation, populationSizeBytes);
    if (err != hipSuccess) { std::cerr << "Error allocating d_newPopulation: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_fitness, fitnessSizeBytes);
    if (err != hipSuccess) { std::cerr << "Error allocating d_fitness: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_parents, parentsSizeBytes);
    if (err != hipSuccess) { std::cerr << "Error allocating d_parents: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_state, stateSizeBytes);
    if (err != hipSuccess) { std::cerr << "Error allocating d_state: " << hipGetErrorString(err) << std::endl; return -1; }

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_state, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error in initializePopulation: " << hipGetErrorString(err) << std::endl; return -1; }

    // Main genetic algorithm loop
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in evaluateFitness: " << hipGetErrorString(err) << std::endl; return -1; }

        // Selection
        selection<<<numBlocks, blockSize>>>(d_fitness, d_parents);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in selection: " << hipGetErrorString(err) << std::endl; return -1; }

        // Crossover
        crossover<<<numBlocks, blockSize>>>(d_population, d_parents, d_newPopulation);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in crossover: " << hipGetErrorString(err) << std::endl; return -1; }

        // Mutation
        mutation<<<numBlocks, blockSize>>>(d_newPopulation, d_state);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in mutation: " << hipGetErrorString(err) << std::endl; return -1; }

        // Swap populations
        float *temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_fitness);
    hipFree(d_parents);
    hipFree(d_state);

    return 0;
}