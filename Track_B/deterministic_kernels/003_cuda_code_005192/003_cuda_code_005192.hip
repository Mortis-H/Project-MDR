// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 1024
#define CHROMOSOME_LENGTH 32
#define GENERATIONS 1000

// Kernel to initialize random chromosomes
__global__ void initializePopulation(hiprandState *states, unsigned int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &states[idx]);
        unsigned int chromosome = 0;
        for (int i = 0; i < CHROMOSOME_LENGTH; i++) {
            chromosome = (chromosome << 1) | (hiprand(&states[idx]) & 1);
        }
        population[idx] = chromosome;
    }
}

// Kernel to evaluate fitness of each chromosome
__global__ void evaluateFitness(unsigned int *population, float *fitness) {
    extern __shared__ unsigned int shared_population[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < POPULATION_SIZE) {
        shared_population[tid] = population[idx];
        __syncthreads();

        // Simple fitness function: count number of 1s
        int ones_count = 0;
        for (int i = 0; i < CHROMOSOME_LENGTH; i++) {
            ones_count += (shared_population[tid] >> i) & 1;
        }
        fitness[idx] = static_cast<float>(ones_count);
    }
}

// Kernel to perform crossover and mutation
__global__ void crossoverAndMutate(hiprandState *states, unsigned int *population, float *fitness) {
    extern __shared__ unsigned int shared_population[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < POPULATION_SIZE) {
        shared_population[tid] = population[idx];
        __syncthreads();

        // Tournament selection: select two parents
        int parent1_idx = tid;
        int parent2_idx = (tid + blockDim.x / 2) % blockDim.x;
        if (fitness[shared_population[parent1_idx]] < fitness[shared_population[parent2_idx]]) {
            parent1_idx = parent2_idx;
        }

        // Crossover
        unsigned int parent1 = shared_population[parent1_idx];
        unsigned int parent2 = shared_population[parent2_idx];
        unsigned int mask = hiprand(&states[idx]) & ((1 << CHROMOSOME_LENGTH) - 1);
        unsigned int child = (parent1 & mask) | (parent2 & ~mask);

        // Mutation
        if (hiprand(&states[idx]) < 0.01f * UINT_MAX) {
            int mutation_pos = hiprand(&states[idx]) % CHROMOSOME_LENGTH;
            child ^= (1 << mutation_pos);
        }

        population[idx] = child;
    }
}

int main() {
    unsigned int *d_population;
    float *d_fitness;
    hiprandState *d_states;

    // Allocate memory on device
    hipError_t err = hipMalloc((void**)&d_population, POPULATION_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating fitness: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Genetic algorithm loop
    for (int generation = 0; generation < GENERATIONS; generation++) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize, blockSize * sizeof(unsigned int)>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error evaluating fitness: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Crossover and mutate
        crossoverAndMutate<<<numBlocks, blockSize, blockSize * sizeof(unsigned int)>>>(d_states, d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in crossover and mutation: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);

    return 0;
}