#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 2048
#define GENES_PER_INDIVIDUAL 1024
#define NUM_GENERATIONS 100

__global__ void initializePopulation(hiprandState *state, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &state[idx]);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            population[idx * GENES_PER_INDIVIDUAL + i] = hiprand_uniform(&state[idx]);
        }
    }
}

__global__ void evaluateFitness(float *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        float fit = 0.0f;
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            fit += population[idx * GENES_PER_INDIVIDUAL + i];
        }
        fitness[idx] = fit;
    }
}

__global__ void selectParents(float *fitness, int *parents) {
    extern __shared__ float shared_fitness[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < POPULATION_SIZE) {
        shared_fitness[tid] = fitness[idx];
        __syncthreads();

        // Simple selection: select the first half as parents
        if (tid < POPULATION_SIZE / 2) {
            parents[idx] = idx;
        }
    }
}

__global__ void crossoverAndMutate(hiprandState *state, float *population, int *parents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = state[idx];
        int parent1 = parents[idx];
        int parent2 = parents[(idx + 1) % POPULATION_SIZE];

        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            float gene1 = population[parent1 * GENES_PER_INDIVIDUAL + i];
            float gene2 = population[parent2 * GENES_PER_INDIVIDUAL + i];
            float newGene = (hiprand_uniform(&localState) > 0.5f) ? gene1 : gene2;
            if (hiprand_uniform(&localState) < 0.01f) { // Mutation rate
                newGene = hiprand_uniform(&localState);
            }
            population[idx * GENES_PER_INDIVIDUAL + i] = newGene;
        }
        state[idx] = localState;
    }
}

int main() {
    float *d_population, *d_fitness;
    hiprandState *d_state;
    int *d_parents;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_population, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating population memory: " << hipGetErrorString(err) << std::endl; return 1; }
    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating fitness memory: " << hipGetErrorString(err) << std::endl; return 1; }
    err = hipMalloc((void**)&d_state, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "Error allocating state memory: " << hipGetErrorString(err) << std::endl; return 1; }
    err = hipMalloc((void**)&d_parents, POPULATION_SIZE * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating parents memory: " << hipGetErrorString(err) << std::endl; return 1; }

    // Initialize population
    int blockSize = 256;
    int numBlocks = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initializePopulation<<<numBlocks, blockSize>>>(d_state, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error in initializePopulation: " << hipGetErrorString(err) << std::endl; return 1; }

    // Genetic Algorithm loop
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        evaluateFitness<<<numBlocks, blockSize>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in evaluateFitness: " << hipGetErrorString(err) << std::endl; return 1; }

        selectParents<<<numBlocks, blockSize>>>(d_fitness, d_parents);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in selectParents: " << hipGetErrorString(err) << std::endl; return 1; }

        crossoverAndMutate<<<numBlocks, blockSize>>>(d_state, d_population, d_parents);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error in crossoverAndMutate: " << hipGetErrorString(err) << std::endl; return 1; }
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_state);
    hipFree(d_parents);

    return 0;
}