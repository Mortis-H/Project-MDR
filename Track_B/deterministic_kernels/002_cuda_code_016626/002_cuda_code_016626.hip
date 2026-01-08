#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define BLOOM_FILTER_SIZE 64
#define NUM_HASH_FUNCTIONS 3

__device__ unsigned int hash1(unsigned int x) {
    x = (x ^ 61) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb2d;
    x = x ^ (x >> 15);
    return x % BLOOM_FILTER_SIZE;
}

__device__ unsigned int hash2(unsigned int x) {
    x = (x + 0x7ed55d16) + (x << 12);
    x = (x ^ 0xc761c23c) ^ (x >> 19);
    x = x + (x << 4);
    x = (x ^ 0x165667b1) ^ (x >> 14);
    return x % BLOOM_FILTER_SIZE;
}

__device__ unsigned int hash3(unsigned int x) {
    x = ~x + (x << 15); // x = (x << 15) - x - 1;
    x = x ^ (x >> 12);
    x = x + (x << 2);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb2d;
    x = x ^ (x >> 16);
    return x % BLOOM_FILTER_SIZE;
}

__device__ void add_to_bloom_filter(unsigned int* bloom_filter, unsigned int element) {
    unsigned int hash_values[NUM_HASH_FUNCTIONS];
    hash_values[0] = hash1(element);
    hash_values[1] = hash2(element);
    hash_values[2] = hash3(element);

    for (int i = 0; i < NUM_HASH_FUNCTIONS; ++i) {
        atomicOr(bloom_filter + hash_values[i], 1);
    }
}

__device__ bool check_in_bloom_filter(unsigned int* bloom_filter, unsigned int element) {
    unsigned int hash_values[NUM_HASH_FUNCTIONS];
    hash_values[0] = hash1(element);
    hash_values[1] = hash2(element);
    hash_values[2] = hash3(element);

    for (int i = 0; i < NUM_HASH_FUNCTIONS; ++i) {
        if (!(bloom_filter[hash_values[i]] & 1)) {
            return false;
        }
    }
    return true;
}

__global__ void insert_elements(unsigned int* bloom_filter, unsigned int* elements, int num_elements) {
    cg::thread_block cta = cg::this_thread_block();
    int tid = cta.thread_rank();
    int stride = cta.size();

    for (int i = tid; i < num_elements; i += stride) {
        add_to_bloom_filter(bloom_filter, elements[i]);
    }
}

__global__ void check_elements(unsigned int* bloom_filter, unsigned int* elements, bool* results, int num_elements) {
    cg::thread_block cta = cg::this_thread_block();
    int tid = cta.thread_rank();
    int stride = cta.size();

    for (int i = tid; i < num_elements; i += stride) {
        results[i] = check_in_bloom_filter(bloom_filter, elements[i]);
    }
}

int main() {
    const int num_elements = 10;
    unsigned int elements[num_elements] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    unsigned int* d_bloom_filter;
    unsigned int* d_elements;
    bool* d_results;

    hipError_t err;

    // Allocate device memory
    err = hipMalloc((void**)&d_bloom_filter, BLOOM_FILTER_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for bloom filter (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    err = hipMalloc((void**)&d_elements, num_elements * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for elements (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    err = hipMalloc((void**)&d_results, num_elements * sizeof(bool));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for results (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize bloom filter to 0
    err = hipMemset(d_bloom_filter, 0, BLOOM_FILTER_SIZE * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize bloom filter (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy elements to device
    err = hipMemcpy(d_elements, elements, num_elements * sizeof(unsigned int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy elements to device (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Insert elements into bloom filter
    insert_elements<<<1, 256>>>(d_bloom_filter, d_elements, num_elements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert_elements kernel (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Check elements in bloom filter
    check_elements<<<1, 256>>>(d_bloom_filter, d_elements, d_results, num_elements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch check_elements kernel (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy results back to host
    bool results[num_elements];
    err = hipMemcpy(results, d_results, num_elements * sizeof(bool), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Print results
    for (int i = 0; i < num_elements; ++i) {
        std::cout << "Element " << elements[i] << " is " << (results[i] ? "probably in" : "definitely not in") << " the bloom filter." << std::endl;
    }

    // Free device memory
    hipFree(d_bloom_filter);
    hipFree(d_elements);
    hipFree(d_results);

    return 0;
}