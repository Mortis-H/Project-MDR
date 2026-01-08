// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 8192
#define GENES_PER_INDIVIDUAL 128
#define MUTATION_RATE 0.01f
#define BLOCK_SIZE 256

__global__ void initializePopulation(hiprandState *state, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE * GENES_PER_INDIVIDUAL) {
        hiprandState localState = state[idx];
        float gene = hiprand_uniform(&localState);
        population[idx] = gene;
        state[idx] = localState;
    }
}

__device__ float fitnessFunction(const float *individual) {
    float fitness = 0.0f;
    for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
        fitness += individual[i];
    }
    return fitness;
}

__global__ void evaluateFitness(const float *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        fitness[idx] = fitnessFunction(&population[idx * GENES_PER_INDIVIDUAL]);
    }
}

__global__ void crossoverAndMutate(hiprandState *state, float *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = state[idx];
        int partnerIdx = hiprand(&localState) % POPULATION_SIZE;
        while (partnerIdx == idx) {
            partnerIdx = hiprand(&localState) % POPULATION_SIZE;
        }

        float crossoverPoint = hiprand_uniform(&localState) * GENES_PER_INDIVIDUAL;
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            float gene;
            if (i < crossoverPoint) {
                gene = population[idx * GENES_PER_INDIVIDUAL + i];
            } else {
                gene = population[partnerIdx * GENES_PER_INDIVIDUAL + i];
            }

            if (hiprand_uniform(&localState) < MUTATION_RATE) {
                gene = hiprand_uniform(&localState);
            }

            population[idx * GENES_PER_INDIVIDUAL + i] = gene;
        }

        fitness[idx] = fitnessFunction(&population[idx * GENES_PER_INDIVIDUAL]);
        state[idx] = localState;
    }
}

int main() {
    float *d_population, *d_fitness;
    hiprandState *d_state;
    int numBytesPopulation = POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float);
    int numBytesFitness = POPULATION_SIZE * sizeof(float);
    int numBytesState = POPULATION_SIZE * sizeof(hiprandState);

    hipError_t err;

    // Allocate memory on the device
    err = hipMalloc((void **)&d_population, numBytesPopulation);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_fitness, numBytesFitness);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for fitness: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_state, numBytesState);
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for state: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize random number states
    unsigned long seed = 1234;
    int numBlocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializePopulation<<<numBlocks, BLOCK_SIZE>>>(d_state, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Evaluate initial fitness
    evaluateFitness<<<numBlocks, BLOCK_SIZE>>>(d_population, d_fitness);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error evaluating initial fitness: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform a single generation of crossover and mutation
    crossoverAndMutate<<<numBlocks, BLOCK_SIZE>>>(d_state, d_population, d_fitness);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error performing crossover and mutation: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_state);

    return 0;
}