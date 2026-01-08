#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_fp16.h>

#define POPULATION_SIZE 1024
#define NUM_VARIABLES 128
#define NUM_GENERATIONS 100
#define BLOCK_SIZE 256

__global__ void initializePopulation(hiprandState *state, float *population, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(seed, idx, 0, &state[idx]);
        for (int i = 0; i < NUM_VARIABLES; ++i) {
            population[idx * NUM_VARIABLES + i] = hiprand_uniform(&state[idx]) * 2.0f - 1.0f;
        }
    }
}

__global__ void evaluateFitness(float *population, float *fitness, int *parentIndices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < NUM_VARIABLES; ++i) {
            float val = population[idx * NUM_VARIABLES + i];
            sum += val * val;
        }
        fitness[idx] = 1.0f / (sum + 1.0f);
        parentIndices[idx] = idx;
    }
}

__global__ void sortIndices(float *fitness, int *parentIndices, int n) {
    extern __shared__ float s_fitness[];
    extern __shared__ int s_indices[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        s_fitness[tid] = fitness[idx];
        s_indices[tid] = parentIndices[idx];
    } else {
        s_fitness[tid] = 0.0f;
        s_indices[tid] = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_fitness[tid] < s_fitness[tid + s]) {
                float tempF = s_fitness[tid];
                s_fitness[tid] = s_fitness[tid + s];
                s_fitness[tid + s] = tempF;

                int tempI = s_indices[tid];
                s_indices[tid] = s_indices[tid + s];
                s_indices[tid + s] = tempI;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        fitness[blockIdx.x] = s_fitness[0];
        parentIndices[blockIdx.x] = s_indices[0];
    }
}

__global__ void selectParents(int *parentIndices, int *selectedParents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = parentIndices[idx % (POPULATION_SIZE / 2)];
        int parent2 = parentIndices[(idx + 1) % (POPULATION_SIZE / 2)];
        selectedParents[idx * 2] = parent1;
        selectedParents[idx * 2 + 1] = parent2;
    }
}

__global__ void crossoverAndMutate(hiprandState *state, float *population, int *selectedParents, float mutationRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = selectedParents[idx * 2];
        int parent2 = selectedParents[idx * 2 + 1];
        for (int i = 0; i < NUM_VARIABLES; ++i) {
            float val1 = population[parent1 * NUM_VARIABLES + i];
            float val2 = population[parent2 * NUM_VARIABLES + i];
            float crossover = hiprand_uniform(&state[idx]);
            float val = crossover < 0.5f ? val1 : val2;
            float mutation = hiprand_uniform(&state[idx]);
            if (mutation < mutationRate) {
                val += (hiprand_uniform(&state[idx]) * 2.0f - 1.0f) * 0.1f;
            }
            population[idx * NUM_VARIABLES + i] = val;
        }
    }
}

void geneticAlgorithm(float *population, float *fitness, int *parentIndices, int *selectedParents, hiprandState *state) {
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        evaluateFitness<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(population, fitness, parentIndices);
        hipDeviceSynchronize();

        int sharedMemSize = (BLOCK_SIZE + 1) * (sizeof(float) + sizeof(int));
        sortIndices<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, sharedMemSize>>>(fitness, parentIndices, POPULATION_SIZE);
        hipDeviceSynchronize();

        selectParents<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(parentIndices, selectedParents);
        hipDeviceSynchronize();

        crossoverAndMutate<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(state, population, selectedParents, 0.01f);
        hipDeviceSynchronize();
    }
}

int main() {
    float *population, *fitness;
    int *parentIndices, *selectedParents;
    hiprandState *state;

    size_t populationSize = POPULATION_SIZE * NUM_VARIABLES * sizeof(float);
    size_t fitnessSize = POPULATION_SIZE * sizeof(float);
    size_t indicesSize = POPULATION_SIZE * sizeof(int);
    size_t stateSize = POPULATION_SIZE * sizeof(hiprandState);

    hipError_t err = hipMalloc(&population, populationSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&fitness, fitnessSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating fitness: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&parentIndices, indicesSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating parentIndices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&selectedParents, indicesSize * 2);
    if (err != hipSuccess) {
        std::cerr << "Error allocating selectedParents: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&state, stateSize);
    if (err != hipSuccess) {
        std::cerr << "Error allocating state: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    initializePopulation<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(state, population, 1234);
    hipDeviceSynchronize();

    geneticAlgorithm(population, fitness, parentIndices, selectedParents, state);

    float *h_population = new float[POPULATION_SIZE * NUM_VARIABLES];
    err = hipMemcpy(h_population, population, populationSize, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying population back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Best solution in the last generation:" << std::endl;
    for (int i = 0; i < NUM_VARIABLES; ++i) {
        std::cout << h_population[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_population;

    hipFree(population);
    hipFree(fitness);
    hipFree(parentIndices);
    hipFree(selectedParents);
    hipFree(state);

    return 0;
}