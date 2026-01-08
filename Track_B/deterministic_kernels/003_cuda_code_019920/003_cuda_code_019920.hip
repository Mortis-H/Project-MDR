#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_PARTICLES 256
#define NUM_BUCKETS 256
#define NUM_GPUS 2

__global__ void countParticles(int* d_input, int* d_counts, int numParticles) {
    extern __shared__ int s_counts[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory
    if (idx < NUM_BUCKETS) {
        s_counts[idx] = 0;
    }
    __syncthreads();

    // Count particles in each bucket
    if (idx < numParticles) {
        atomicAdd(&s_counts[d_input[idx]], 1);
    }
    __syncthreads();

    // Write counts to global memory
    if (idx < NUM_BUCKETS) {
        atomicAdd(&d_counts[idx], s_counts[idx]);
    }
}

__global__ void scanCounts(int* d_counts, int numBuckets) {
    extern __shared__ int s_counts[];

    int idx = threadIdx.x;
    int stride;

    // Load data into shared memory
    if (idx < numBuckets) {
        s_counts[idx] = d_counts[idx];
    } else {
        s_counts[idx] = 0;
    }
    __syncthreads();

    // Upsweep phase
    for (stride = 1; stride < numBuckets; stride *= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < numBuckets) {
            s_counts[index] += s_counts[index - stride];
        }
        __syncthreads();
    }

    // Clear the last element
    if (threadIdx.x == 0) {
        s_counts[numBuckets - 1] = 0;
    }
    __syncthreads();

    // Downsweep phase
    for (stride = numBuckets / 2; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < numBuckets) {
            int temp = s_counts[index];
            s_counts[index] += s_counts[index - stride];
            s_counts[index - stride] = temp;
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (idx < numBuckets) {
        d_counts[idx] = s_counts[idx];
    }
}

__global__ void sortParticles(int* d_input, int* d_output, int* d_counts, int numParticles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numParticles) {
        int value = d_input[idx];
        int position = atomicAdd(&d_counts[value], 1);
        d_output[position] = value;
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int h_input[NUM_PARTICLES];
    int h_output[NUM_PARTICLES];
    int h_counts[NUM_BUCKETS] = {0};

    // Initialize input data
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_input[i] = rand() % NUM_BUCKETS;
    }

    int* d_input[NUM_GPUS];
    int* d_output[NUM_GPUS];
    int* d_counts[NUM_GPUS];
    int* h_counts_all = new int[NUM_BUCKETS * NUM_GPUS];

    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);

        // Allocate device memory
        checkCudaError(hipMalloc(&d_input[gpu], NUM_PARTICLES * sizeof(int)), "hipMalloc d_input");
        checkCudaError(hipMalloc(&d_output[gpu], NUM_PARTICLES * sizeof(int)), "hipMalloc d_output");
        checkCudaError(hipMalloc(&d_counts[gpu], NUM_BUCKETS * sizeof(int)), "hipMalloc d_counts");

        // Copy input data to device
        checkCudaError(hipMemcpy(d_input[gpu], h_input, NUM_PARTICLES * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy d_input");

        // Initialize counts to zero on device
        checkCudaError(hipMemset(d_counts[gpu], 0, NUM_BUCKETS * sizeof(int)), "hipMemset d_counts");

        // Launch countParticles kernel
        countParticles<<<(NUM_PARTICLES + 255) / 256, 256, NUM_BUCKETS * sizeof(int)>>>(d_input[gpu], d_counts[gpu], NUM_PARTICLES);
        checkCudaError(hipGetLastError(), "countParticles kernel launch");

        // Copy counts back to host
        checkCudaError(hipMemcpy(&h_counts_all[gpu * NUM_BUCKETS], d_counts[gpu], NUM_BUCKETS * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d_counts");
    }

    // Sum counts across all GPUs
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        h_counts[i] = 0;
        for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
            h_counts[i] += h_counts_all[gpu * NUM_BUCKETS + i];
        }
    }

    // Perform scan on the summed counts
    int* d_counts_all;
    checkCudaError(hipMalloc(&d_counts_all, NUM_BUCKETS * sizeof(int)), "hipMalloc d_counts_all");
    checkCudaError(hipMemcpy(d_counts_all, h_counts, NUM_BUCKETS * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy d_counts_all");
    scanCounts<<<1, NUM_BUCKETS, NUM_BUCKETS * sizeof(int)>>>(d_counts_all, NUM_BUCKETS);
    checkCudaError(hipGetLastError(), "scanCounts kernel launch");
    checkCudaError(hipMemcpy(h_counts, d_counts_all, NUM_BUCKETS * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d_counts_all");

    // Copy scan results back to device counts
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        checkCudaError(hipMemcpy(d_counts[gpu], h_counts, NUM_BUCKETS * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy d_counts");
    }

    // Launch sortParticles kernel
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        sortParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_input[gpu], d_output[gpu], d_counts[gpu], NUM_PARTICLES);
        checkCudaError(hipGetLastError(), "sortParticles kernel launch");
    }

    // Copy sorted data back to host
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        checkCudaError(hipMemcpy(&h_output[gpu * NUM_PARTICLES / NUM_GPUS], d_output[gpu], NUM_PARTICLES / NUM_GPUS * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d_output");
    }

    // Verify the result
    for (int i = 1; i < NUM_PARTICLES; ++i) {
        if (h_output[i - 1] > h_output[i]) {
            std::cerr << "Sorting failed!" << std::endl;
            return -1;
        }
    }

    std::cout << "Sorting successful!" << std::endl;

    // Free device memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(d_input[gpu]);
        hipFree(d_output[gpu]);
        hipFree(d_counts[gpu]);
    }
    hipFree(d_counts_all);

    delete[] h_counts_all;

    return 0;
}