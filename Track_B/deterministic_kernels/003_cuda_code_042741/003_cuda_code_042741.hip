#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 128
#define GENOME_LENGTH 32
#define NUM_GENERATIONS 1000

// Kernel to initialize random genomes
__global__ void initGenomes(hiprandState *states, unsigned int *genomes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            genomes[idx * GENOME_LENGTH + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness of genomes (simple example: count number of 1s)
__global__ void evaluateFitness(unsigned int *genomes, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int count = 0;
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            count += genomes[idx * GENOME_LENGTH + i];
        }
        fitness[idx] = count;
    }
}

// Kernel to perform selection, crossover, and mutation
__global__ void evolvePopulation(unsigned int *genomes, hiprandState *states) {
    extern __shared__ unsigned int sharedGenomes[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        int parent1 = idx;
        int parent2 = hiprand(&localState) % POPULATION_SIZE;
        int crossoverPoint = hiprand(&localState) % GENOME_LENGTH;

        // Crossover
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            if (i < crossoverPoint) {
                sharedGenomes[idx * GENOME_LENGTH + i] = genomes[parent1 * GENOME_LENGTH + i];
            } else {
                sharedGenomes[idx * GENOME_LENGTH + i] = genomes[parent2 * GENOME_LENGTH + i];
            }
        }

        // Mutation
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            if (hiprand(&localState) < 0.01f) { // 1% mutation rate
                sharedGenomes[idx * GENOME_LENGTH + i] = 1 - sharedGenomes[idx * GENOME_LENGTH + i];
            }
        }

        // Write back to global memory
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            genomes[idx * GENOME_LENGTH + i] = sharedGenomes[idx * GENOME_LENGTH + i];
        }
        states[idx] = localState;
    }
}

int main() {
    unsigned int *d_genomes;
    float *d_fitness;
    hiprandState *d_states;
    unsigned int *h_genomes = new unsigned int[POPULATION_SIZE * GENOME_LENGTH];
    float *h_fitness = new float[POPULATION_SIZE];

    // Allocate device memory
    hipError_t err = hipMalloc(&d_genomes, POPULATION_SIZE * GENOME_LENGTH * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for genomes: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for fitness: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize genomes asynchronously
    hipStream_t stream;
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "Error creating stream: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initGenomes<<<numBlocks, blockSize, 0, stream>>>(d_states, d_genomes);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initGenomes kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Main loop of genetic algorithm
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<numBlocks, blockSize, 0, stream>>>(d_genomes, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching evaluateFitness kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Evolve population
        evolvePopulation<<<numBlocks, blockSize, POPULATION_SIZE * GENOME_LENGTH * sizeof(unsigned int), stream>>>(d_genomes, d_states);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching evolvePopulation kernel: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy results back to host
    err = hipMemcpyAsync(h_fitness, d_fitness, POPULATION_SIZE * sizeof(float), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        std::cerr << "Error copying fitness data back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize and clean up
    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        std::cerr << "Error synchronizing stream: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipFree(d_genomes);
    hipFree(d_fitness);
    hipFree(d_states);
    hipStreamDestroy(stream);
    delete[] h_genomes;
    delete[] h_fitness;

    std::cout << "Genetic algorithm completed successfully." << std::endl;
    return 0;
}