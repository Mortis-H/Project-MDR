// genetic_algorithm.cu
// A simple CUDA implementation of a genetic algorithm with dynamic parallelism
// optimized for Turing architecture.

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 256
#define GENOME_SIZE 16

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENOME_SIZE; ++i) {
            population[idx * GENOME_SIZE + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness of each genome
__global__ void evaluateFitness(int *population, int *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int fit = 0;
        for (int i = 0; i < GENOME_SIZE; ++i) {
            fit += population[idx * GENOME_SIZE + i];
        }
        fitness[idx] = fit;
    }
}

// Kernel to perform selection based on fitness
__global__ void selection(int *population, int *fitness, int *newPopulation) {
    extern __shared__ int s_fitness[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Load fitness into shared memory
        s_fitness[threadIdx.x] = fitness[idx];
        __syncthreads();

        // Simple selection: choose the best half
        int bestIdx = 0;
        for (int i = 1; i < blockDim.x; ++i) {
            if (s_fitness[i] > s_fitness[bestIdx]) {
                bestIdx = i;
            }
        }
        if (threadIdx.x == bestIdx) {
            int globalIdx = blockIdx.x * blockDim.x + bestIdx;
            for (int i = 0; i < GENOME_SIZE; ++i) {
                newPopulation[idx * GENOME_SIZE + i] = population[globalIdx * GENOME_SIZE + i];
            }
        }
    }
}

// Kernel to perform crossover
__global__ void crossover(int *population, int *newPopulation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE / 2) {
        int parent1 = idx;
        int parent2 = POPULATION_SIZE - idx - 1;
        int crossoverPoint = GENOME_SIZE / 2;
        for (int i = 0; i < crossoverPoint; ++i) {
            newPopulation[idx * GENOME_SIZE + i] = population[parent1 * GENOME_SIZE + i];
            newPopulation[(POPULATION_SIZE - idx - 1) * GENOME_SIZE + i] = population[parent2 * GENOME_SIZE + i];
        }
        for (int i = crossoverPoint; i < GENOME_SIZE; ++i) {
            newPopulation[idx * GENOME_SIZE + i] = population[parent2 * GENOME_SIZE + i];
            newPopulation[(POPULATION_SIZE - idx - 1) * GENOME_SIZE + i] = population[parent1 * GENOME_SIZE + i];
        }
    }
}

// Kernel to perform mutation
__global__ void mutate(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        for (int i = 0; i < GENOME_SIZE; ++i) {
            if (hiprand(&localState) % 100 < 5) { // 5% mutation rate
                population[idx * GENOME_SIZE + i] = 1 - population[idx * GENOME_SIZE + i];
            }
        }
        states[idx] = localState;
    }
}

int main() {
    int *d_population, *d_newPopulation, *d_fitness;
    hiprandState *d_states;

    // Allocate memory on device
    hipError_t err = hipMalloc((void**)&d_population, POPULATION_SIZE * GENOME_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for population: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void**)&d_newPopulation, POPULATION_SIZE * GENOME_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for new population: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for fitness: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc((void**)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for states: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing population: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Perform genetic algorithm steps
    for (int generation = 0; generation < 10; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error evaluating fitness: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        // Selection
        selection<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_population, d_fitness, d_newPopulation);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error selecting population: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        // Crossover
        crossover<<<numBlocks / 2, blockSize>>>(d_newPopulation, d_population);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error performing crossover: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        // Mutate
        mutate<<<numBlocks, blockSize>>>(d_states, d_population);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error mutating population: " << hipGetErrorString(err) << std::endl;
            return 1;
        }
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_fitness);
    hipFree(d_states);

    return 0;
}