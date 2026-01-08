#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#define NUM_PARTICLES 256
#define BUCKET_SIZE 32
#define NUM_BUCKETS (NUM_PARTICLES / BUCKET_SIZE)

// Kernel to distribute particles into buckets
__global__ void distributeParticles(const float3* particles, int* bucket_indices, int* bucket_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    // Simple bucket index based on the x-coordinate (normalized to [0, NUM_BUCKETS-1])
    int bucket = static_cast<int>(particles[idx].x * NUM_BUCKETS);
    bucket = min(bucket, NUM_BUCKETS - 1); // Ensure bucket index is within bounds

    // Store the bucket index for each particle
    bucket_indices[idx] = bucket;

    // Atomic increment to count particles in each bucket
    atomicAdd(&bucket_counts[bucket], 1);
}

// Kernel to sort particles within each bucket
__global__ void sortBuckets(float3* particles, int* bucket_indices, int* bucket_offsets) {
    __shared__ float3 shared_particles[BUCKET_SIZE];
    __shared__ int shared_indices[BUCKET_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    // Load particle and its bucket index into shared memory
    shared_particles[threadIdx.x] = particles[idx];
    shared_indices[threadIdx.x] = bucket_indices[idx];

    __syncthreads();

    // Sort within each bucket using shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int left_idx = bucket_offsets[shared_indices[threadIdx.x]] + threadIdx.x;
            int right_idx = left_idx + stride;

            if (right_idx < NUM_PARTICLES && shared_particles[threadIdx.x].x > shared_particles[threadIdx.x + stride].x) {
                float3 temp = shared_particles[threadIdx.x];
                shared_particles[threadIdx.x] = shared_particles[threadIdx.x + stride];
                shared_particles[threadIdx.x + stride] = temp;
            }
        }
        __syncthreads();
    }

    // Write sorted particles back to global memory
    int global_idx = bucket_offsets[shared_indices[threadIdx.x]] + threadIdx.x;
    particles[global_idx] = shared_particles[threadIdx.x];
}

int main() {
    // Initialize host particles with random positions in [0, 1)
    thrust::host_vector<float3> h_particles(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_particles[i] = make_float3(static_cast<float>(rand()) / RAND_MAX,
                                     static_cast<float>(rand()) / RAND_MAX,
                                     static_cast<float>(rand()) / RAND_MAX);
    }

    // Allocate device memory
    thrust::device_vector<float3> d_particles = h_particles;
    thrust::device_vector<int> d_bucket_indices(NUM_PARTICLES);
    thrust::device_vector<int> d_bucket_counts(NUM_BUCKETS, 0);
    thrust::device_vector<int> d_bucket_offsets(NUM_BUCKETS);

    // Distribute particles into buckets
    int block_size = 256;
    int grid_size = (NUM_PARTICLES + block_size - 1) / block_size;
    distributeParticles<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_particles.data()),
                                                  thrust::raw_pointer_cast(d_bucket_indices.data()),
                                                  thrust::raw_pointer_cast(d_bucket_counts.data()));
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in distributeParticles: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Compute bucket offsets
    thrust::exclusive_scan(thrust::device, d_bucket_counts.begin(), d_bucket_counts.end(), d_bucket_offsets.begin());

    // Sort particles within each bucket
    sortBuckets<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_particles.data()),
                                           thrust::raw_pointer_cast(d_bucket_indices.data()),
                                           thrust::raw_pointer_cast(d_bucket_offsets.data()));
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in sortBuckets: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy sorted particles back to host
    thrust::host_vector<float3> h_sorted_particles = d_particles;

    // Output sorted particles
    std::cout << "Sorted particles:" << std::endl;
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        std::cout << "(" << h_sorted_particles[i].x << ", " << h_sorted_particles[i].y << ", " << h_sorted_particles[i].z << ")" << std::endl;
    }

    return 0;
}