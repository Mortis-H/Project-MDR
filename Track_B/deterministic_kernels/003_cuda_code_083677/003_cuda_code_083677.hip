#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARTICLES 1024
#define NUM_DIMENSIONS 1024
#define NUM_ITERATIONS 1000

__constant__ float c_global_best[NUM_DIMENSIONS];

__global__ void initializeParticles(float *positions, float *velocities, float *personal_best_positions, float *personal_best_values, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        for (int d = 0; d < NUM_DIMENSIONS; ++d) {
            positions[idx * NUM_DIMENSIONS + d] = hiprand_uniform(&localState) * 100.0f; // Random position
            velocities[idx * NUM_DIMENSIONS + d] = hiprand_uniform(&localState) * 10.0f - 5.0f; // Random velocity
            personal_best_positions[idx * NUM_DIMENSIONS + d] = positions[idx * NUM_DIMENSIONS + d];
        }
        personal_best_values[idx] = 1e9f; // Initialize to a large value
        states[idx] = localState;
    }
}

__device__ float evaluateObjectiveFunction(float *position) {
    float sum = 0.0f;
    for (int d = 0; d < NUM_DIMENSIONS; ++d) {
        sum += position[d] * position[d]; // Sphere function
    }
    return sum;
}

__global__ void updateParticles(float *positions, float *velocities, float *personal_best_positions, float *personal_best_values, float w, float c1, float c2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        float current_value = evaluateObjectiveFunction(&positions[idx * NUM_DIMENSIONS]);
        if (current_value < personal_best_values[idx]) {
            personal_best_values[idx] = current_value;
            for (int d = 0; d < NUM_DIMENSIONS; ++d) {
                personal_best_positions[idx * NUM_DIMENSIONS + d] = positions[idx * NUM_DIMENSIONS + d];
            }
        }

        hiprandState localState;
        hiprand_init(clock64(), idx, 0, &localState);

        for (int d = 0; d < NUM_DIMENSIONS; ++d) {
            float r1 = hiprand_uniform(&localState);
            float r2 = hiprand_uniform(&localState);
            velocities[idx * NUM_DIMENSIONS + d] = w * velocities[idx * NUM_DIMENSIONS + d] +
                                                  c1 * r1 * (personal_best_positions[idx * NUM_DIMENSIONS + d] - positions[idx * NUM_DIMENSIONS + d]) +
                                                  c2 * r2 * (c_global_best[d] - positions[idx * NUM_DIMENSIONS + d]);
            positions[idx * NUM_DIMENSIONS + d] += velocities[idx * NUM_DIMENSIONS + d];
        }
    }
}

__global__ void findGlobalBest(float *personal_best_positions, float *personal_best_values, float *global_best, float *global_best_value) {
    extern __shared__ float shared_best_values[];
    extern __shared__ float shared_best_positions[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < NUM_PARTICLES) {
        shared_best_values[tid] = personal_best_values[idx];
        shared_best_positions[tid * NUM_DIMENSIONS] = personal_best_positions[idx * NUM_DIMENSIONS];
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (shared_best_values[tid] > shared_best_values[tid + s]) {
                    shared_best_values[tid] = shared_best_values[tid + s];
                    for (int d = 0; d < NUM_DIMENSIONS; ++d) {
                        shared_best_positions[tid * NUM_DIMENSIONS + d] = shared_best_positions[(tid + s) * NUM_DIMENSIONS + d];
                    }
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicMin(reinterpret_cast<unsigned int*>(global_best_value), __float_as_uint(shared_best_values[0]));
            if (__float_as_uint(*global_best_value) == __float_as_uint(shared_best_values[0])) {
                for (int d = 0; d < NUM_DIMENSIONS; ++d) {
                    global_best[d] = shared_best_positions[d];
                }
            }
        }
    }
}

int main() {
    float *d_positions, *d_velocities, *d_personal_best_positions, *d_personal_best_values;
    float *d_global_best, *d_global_best_value;
    hiprandState *d_states;

    hipError_t err;

    // Allocate memory
    err = hipMalloc((void**)&d_positions, NUM_PARTICLES * NUM_DIMENSIONS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for positions\n"; return -1; }
    err = hipMalloc((void**)&d_velocities, NUM_PARTICLES * NUM_DIMENSIONS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for velocities\n"; return -1; }
    err = hipMalloc((void**)&d_personal_best_positions, NUM_PARTICLES * NUM_DIMENSIONS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for personal_best_positions\n"; return -1; }
    err = hipMalloc((void**)&d_personal_best_values, NUM_PARTICLES * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for personal_best_values\n"; return -1; }
    err = hipMalloc((void**)&d_global_best, NUM_DIMENSIONS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for global_best\n"; return -1; }
    err = hipMalloc((void**)&d_global_best_value, sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for global_best_value\n"; return -1; }
    err = hipMalloc((void**)&d_states, NUM_PARTICLES * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for states\n"; return -1; }

    // Initialize particles
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    initializeParticles<<<numBlocks, blockSize>>>(d_positions, d_velocities, d_personal_best_positions, d_personal_best_values, d_states, time(NULL));
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error launching initializeParticles kernel\n"; return -1; }

    // Initialize global best
    float h_global_best[NUM_DIMENSIONS];
    float h_global_best_value = 1e9f;
    hipMemcpy(d_global_best, h_global_best, NUM_DIMENSIONS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_global_best_value, &h_global_best_value, sizeof(float), hipMemcpyHostToDevice);

    // PSO parameters
    float w = 0.729;
    float c1 = 1.49445;
    float c2 = 1.49445;

    // Run PSO iterations
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        updateParticles<<<numBlocks, blockSize>>>(d_positions, d_velocities, d_personal_best_positions, d_personal_best_values, w, c1, c2);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error launching updateParticles kernel\n"; return -1; }

        findGlobalBest<<<numBlocks, blockSize, (blockSize + 1) * NUM_DIMENSIONS * sizeof(float)>>>(d_personal_best_positions, d_personal_best_values, d_global_best, d_global_best_value);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error launching findGlobalBest kernel\n"; return -1; }

        hipMemcpy(h_global_best, d_global_best, NUM_DIMENSIONS * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(&h_global_best_value, d_global_best_value, sizeof(float), hipMemcpyDeviceToHost);
    }

    // Output result
    std::cout << "Global Best Value: " << h_global_best_value << std::endl;
    std::cout << "Global Best Position: ";
    for (int d = 0; d < NUM_DIMENSIONS; ++d) {
        std::cout << h_global_best[d] << " ";
    }
    std::cout << std::endl;

    // Free memory
    hipFree(d_positions);
    hipFree(d_velocities);
    hipFree(d_personal_best_positions);
    hipFree(d_personal_best_values);
    hipFree(d_global_best);
    hipFree(d_global_best_value);
    hipFree(d_states);

    return 0;
}