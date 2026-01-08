#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define BUCKET_SIZE 16
#define NUM_BUCKETS 512
#define WARP_SIZE 32

struct Node {
    int key;
    int value;
    int next;
};

__global__ void insertKernel(Node* d_nodes, int* d_bucket_heads, int* d_bucket_locks, int* d_keys, int* d_values, int num_elements) {
    extern __shared__ Node shared_nodes[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_elements) return;

    int key = d_keys[idx];
    int value = d_values[idx];
    int bucket = key % NUM_BUCKETS;

    // Acquire lock for the bucket
    while (atomicCAS(&d_bucket_locks[bucket], 0, 1) != 0);

    // Load bucket head into shared memory
    int head = d_bucket_heads[bucket];
    int current = head;
    int prev = -1;

    // Traverse the list in shared memory
    while (current != -1) {
        shared_nodes[threadIdx.x] = d_nodes[current];
        __syncthreads();

        if (shared_nodes[threadIdx.x].key == key) {
            // Key already exists, update value
            atomicExch(&d_nodes[current].value, value);
            // Release lock
            atomicExch(&d_bucket_locks[bucket], 0);
            return;
        }

        prev = current;
        current = shared_nodes[threadIdx.x].next;
        __syncthreads();
    }

    // Find an empty slot in shared memory
    int empty_slot = -1;
    for (int i = 0; i < BUCKET_SIZE; i++) {
        if (shared_nodes[i].key == -1) {
            empty_slot = i;
            break;
        }
    }

    if (empty_slot == -1) {
        // No empty slot found, release lock and exit
        atomicExch(&d_bucket_locks[bucket], 0);
        return;
    }

    // Insert new node
    shared_nodes[empty_slot].key = key;
    shared_nodes[empty_slot].value = value;
    shared_nodes[empty_slot].next = -1;

    if (prev == -1) {
        // New head of the list
        d_bucket_heads[bucket] = empty_slot;
    } else {
        // Append to the end of the list
        shared_nodes[prev].next = empty_slot;
    }

    // Write back to global memory
    d_nodes[empty_slot] = shared_nodes[empty_slot];

    // Release lock
    atomicExch(&d_bucket_locks[bucket], 0);
}

__global__ void lookupKernel(Node* d_nodes, int* d_bucket_heads, int* d_bucket_locks, int* d_keys, int* d_results, int num_elements) {
    extern __shared__ Node shared_nodes[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_elements) return;

    int key = d_keys[idx];
    int bucket = key % NUM_BUCKETS;

    // Acquire lock for the bucket
    while (atomicCAS(&d_bucket_locks[bucket], 0, 1) != 0);

    // Load bucket head into shared memory
    int current = d_bucket_heads[bucket];

    // Traverse the list in shared memory
    while (current != -1) {
        shared_nodes[threadIdx.x] = d_nodes[current];
        __syncthreads();

        if (shared_nodes[threadIdx.x].key == key) {
            d_results[idx] = shared_nodes[threadIdx.x].value;
            // Release lock
            atomicExch(&d_bucket_locks[bucket], 0);
            return;
        }

        current = shared_nodes[threadIdx.x].next;
        __syncthreads();
    }

    // Key not found
    d_results[idx] = -1;

    // Release lock
    atomicExch(&d_bucket_locks[bucket], 0);
}

int main() {
    const int num_elements = 1024;
    const int num_nodes = NUM_BUCKETS * BUCKET_SIZE;

    // Initialize host data
    thrust::host_vector<int> h_keys(num_elements);
    thrust::host_vector<int> h_values(num_elements);
    thrust::host_vector<int> h_results(num_elements, -1);
    thrust::host_vector<Node> h_nodes(num_nodes, Node{-1, -1, -1});
    thrust::host_vector<int> h_bucket_heads(NUM_BUCKETS, -1);
    thrust::host_vector<int> h_bucket_locks(NUM_BUCKETS, 0);

    // Fill keys and values with some data
    for (int i = 0; i < num_elements; i++) {
        h_keys[i] = i;
        h_values[i] = i * 2;
    }

    // Copy data to device
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_values = h_values;
    thrust::device_vector<int> d_results = h_results;
    thrust::device_vector<Node> d_nodes = h_nodes;
    thrust::device_vector<int> d_bucket_heads = h_bucket_heads;
    thrust::device_vector<int> d_bucket_locks = h_bucket_locks;

    // Launch insert kernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * sizeof(Node);

    insertKernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_bucket_heads.data()),
        thrust::raw_pointer_cast(d_bucket_locks.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_values.data()),
        num_elements
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in insertKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch lookup kernel
    lookupKernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_bucket_heads.data()),
        thrust::raw_pointer_cast(d_bucket_locks.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_results.data()),
        num_elements
    );

    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in lookupKernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    h_results = d_results;

    // Print results
    for (int i = 0; i < num_elements; i++) {
        if (h_results[i] != h_values[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << h_values[i] << ", got " << h_results[i] << std::endl;
            return -1;
        }
    }

    std::cout << "All lookups successful!" << std::endl;

    return 0;
}