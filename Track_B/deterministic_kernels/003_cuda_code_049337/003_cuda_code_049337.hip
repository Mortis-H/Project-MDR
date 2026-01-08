#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the number of elements in the bloom filter
#define BLOOM_FILTER_SIZE 8192
// Define the number of hash functions
#define NUM_HASH_FUNCTIONS 5

// Kernel to initialize the bloom filter to zero
__global__ void initBloomFilter(unsigned int* bloomFilter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BLOOM_FILTER_SIZE) {
        bloomFilter[idx] = 0;
    }
}

// Kernel to add an element to the bloom filter
__global__ void addToBloomFilter(unsigned int* bloomFilter, unsigned int* elements, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int element = elements[idx];
        for (int i = 0; i < NUM_HASH_FUNCTIONS; ++i) {
            unsigned int hash = (element + i) % BLOOM_FILTER_SIZE;
            atomicOr(&bloomFilter[hash], 1U << (hash % 32));
        }
    }
}

// Kernel to check if an element is in the bloom filter
__global__ void checkBloomFilter(unsigned int* bloomFilter, unsigned int* elements, bool* results, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int element = elements[idx];
        bool isPresent = true;
        for (int i = 0; i < NUM_HASH_FUNCTIONS; ++i) {
            unsigned int hash = (element + i) % BLOOM_FILTER_SIZE;
            if ((bloomFilter[hash] & (1U << (hash % 32))) == 0) {
                isPresent = false;
                break;
            }
        }
        results[idx] = isPresent;
    }
}

int main() {
    // Number of elements to add and check
    int numElements = 1024;

    // Allocate memory for bloom filter on the host
    unsigned int* h_bloomFilter = (unsigned int*)calloc(BLOOM_FILTER_SIZE, sizeof(unsigned int));
    if (h_bloomFilter == nullptr) {
        std::cerr << "Failed to allocate host memory for bloom filter" << std::endl;
        return -1;
    }

    // Allocate memory for elements on the host
    unsigned int* h_elements = (unsigned int*)malloc(numElements * sizeof(unsigned int));
    if (h_elements == nullptr) {
        std::cerr << "Failed to allocate host memory for elements" << std::endl;
        free(h_bloomFilter);
        return -1;
    }

    // Allocate memory for results on the host
    bool* h_results = (bool*)malloc(numElements * sizeof(bool));
    if (h_results == nullptr) {
        std::cerr << "Failed to allocate host memory for results" << std::endl;
        free(h_bloomFilter);
        free(h_elements);
        return -1;
    }

    // Initialize random elements
    hiprandState* d_states;
    hipMalloc(&d_states, numElements * sizeof(hiprandState));
    unsigned int* d_bloomFilter;
    hipMalloc(&d_bloomFilter, BLOOM_FILTER_SIZE * sizeof(unsigned int));
    unsigned int* d_elements;
    hipMalloc(&d_elements, numElements * sizeof(unsigned int));
    bool* d_results;
    hipMalloc(&d_results, numElements * sizeof(bool));

    // Initialize random number generator states
    hiprandState* states = d_states;
    unsigned int seed = 1234;
    unsigned int* elements = h_elements;
    for (int i = 0; i < numElements; ++i) {
        elements[i] = i;
    }

    // Copy elements to device
    hipMemcpy(d_elements, h_elements, numElements * sizeof(unsigned int), hipMemcpyHostToDevice);

    // Initialize bloom filter on device
    int blockSize = 256;
    int numBlocks = (BLOOM_FILTER_SIZE + blockSize - 1) / blockSize;
    initBloomFilter<<<numBlocks, blockSize>>>(d_bloomFilter);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize bloom filter: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Add elements to bloom filter
    numBlocks = (numElements + blockSize - 1) / blockSize;
    addToBloomFilter<<<numBlocks, blockSize>>>(d_bloomFilter, d_elements, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to add elements to bloom filter: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Check elements in bloom filter
    checkBloomFilter<<<numBlocks, blockSize>>>(d_bloomFilter, d_elements, d_results, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to check elements in bloom filter: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_results, d_results, numElements * sizeof(bool), hipMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < numElements; ++i) {
        std::cout << "Element " << i << " is " << (h_results[i] ? "possibly in" : "not in") << " the bloom filter." << std::endl;
    }

    // Free device memory
    hipFree(d_bloomFilter);
    hipFree(d_elements);
    hipFree(d_results);
    hipFree(d_states);

    // Free host memory
    free(h_bloomFilter);
    free(h_elements);
    free(h_results);

    return 0;
}