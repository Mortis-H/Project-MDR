// genetic_algorithm.cu
// A simple genetic algorithm implementation in CUDA for a population size of 512.
// Optimized for compute throughput on Turing architecture.

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Constants
#define POPULATION_SIZE 512
#define GENOME_SIZE 32
#define GENERATIONS 100

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, unsigned int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Initialize the curand state
        hiprand_init(idx, 0, 0, &states[idx]);
        // Generate random genome
        for (int i = 0; i < GENOME_SIZE; ++i) {
            population[idx * GENOME_SIZE + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness (simple example: count number of 1s)
__global__ void evaluateFitness(unsigned int *population, int *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int fit = 0;
        for (int i = 0; i < GENOME_SIZE; ++i) {
            fit += population[idx * GENOME_SIZE + i];
        }
        fitness[idx] = fit;
    }
}

// Kernel to perform selection (simple example: copy the best genome to all)
__global__ void selectBest(unsigned int *population, int *fitness) {
    __shared__ int bestFitness;
    __shared__ int bestIdx;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    if (localIdx == 0) {
        bestFitness = 0;
        bestIdx = 0;
    }
    __syncthreads();

    if (idx < POPULATION_SIZE) {
        // Find the best fitness
        int localBestFitness = fitness[idx];
        int localBestIdx = idx;
        atomicMax(&bestFitness, localBestFitness);
        if (localBestFitness == bestFitness) {
            atomicCAS(&bestIdx, bestIdx, localBestIdx);
        }
    }
    __syncthreads();

    if (idx < POPULATION_SIZE) {
        // Copy the best genome to all
        for (int i = 0; i < GENOME_SIZE; ++i) {
            population[idx * GENOME_SIZE + i] = population[bestIdx * GENOME_SIZE + i];
        }
    }
}

int main() {
    unsigned int *d_population;
    int *d_fitness;
    hiprandState *d_states;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void **)&d_population, POPULATION_SIZE * GENOME_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate population memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_fitness, POPULATION_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate fitness memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        return -1;
    }
    err = hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate states memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        return -1;
    }

    // Initialize the population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializePopulation kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        hipFree(d_states);
        return -1;
    }

    // Main genetic algorithm loop
    for (int gen = 0; gen < GENERATIONS; ++gen) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch evaluateFitness kernel: " << hipGetErrorString(err) << std::endl;
            hipFree(d_population);
            hipFree(d_fitness);
            hipFree(d_states);
            return -1;
        }

        // Selection
        selectBest<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch selectBest kernel: " << hipGetErrorString(err) << std::endl;
            hipFree(d_population);
            hipFree(d_fitness);
            hipFree(d_states);
            return -1;
        }
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);

    std::cout << "Genetic algorithm completed successfully." << std::endl;
    return 0;
}