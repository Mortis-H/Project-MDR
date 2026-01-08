#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_BUCKETS 8192
#define MAX_CHAIN_LENGTH 256

// Kernel to initialize the hash table
__global__ void initHashTable(int* keys, int* next, int* buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_BUCKETS) {
        buckets[idx] = -1; // -1 indicates the end of the chain
    }
}

// Kernel to insert keys into the hash table using separate chaining
__global__ void insertKeys(int* keys, int* next, int* buckets, int numKeys, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState state;
    hiprand_init(seed, idx, 0, &state);

    if (idx < numKeys) {
        int key = hiprand(&state) % (NUM_BUCKETS * 10); // Generate a random key
        keys[idx] = key;
        int bucketIdx = key % NUM_BUCKETS;

        // Insert at the head of the chain
        next[idx] = buckets[bucketIdx];
        buckets[bucketIdx] = idx;
    }
}

// Kernel to search for keys in the hash table
__global__ void searchKeys(int* keys, int* next, int* buckets, int* results, int numSearches, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState state;
    hiprand_init(seed, idx, 0, &state);

    if (idx < numSearches) {
        int searchKey = hiprand(&state) % (NUM_BUCKETS * 10); // Generate a random search key
        int bucketIdx = searchKey % NUM_BUCKETS;
        int current = buckets[bucketIdx];
        bool found = false;

        while (current != -1) {
            if (keys[current] == searchKey) {
                found = true;
                break;
            }
            current = next[current];
        }

        results[idx] = found ? 1 : 0;
    }
}

int main() {
    const int numKeys = 32768; // Number of keys to insert
    const int numSearches = 32768; // Number of keys to search for

    // Allocate memory on the host
    int* h_keys = new int[numKeys];
    int* h_next = new int[numKeys];
    int* h_buckets = new int[NUM_BUCKETS];
    int* h_results = new int[numSearches];

    // Allocate memory on the device
    int* d_keys, *d_next, *d_buckets, *d_results;
    hipError_t err;

    err = hipMalloc(&d_keys, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for keys" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_next, numKeys * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for next" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_buckets, NUM_BUCKETS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for buckets" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_results, numSearches * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for results" << std::endl;
        return -1;
    }

    // Initialize hash table
    int blockSize = 256;
    int numBlocks = (NUM_BUCKETS + blockSize - 1) / blockSize;
    initHashTable<<<numBlocks, blockSize>>>(d_keys, d_next, d_buckets);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initHashTable kernel" << std::endl;
        return -1;
    }

    // Insert keys
    numBlocks = (numKeys + blockSize - 1) / blockSize;
    insertKeys<<<numBlocks, blockSize>>>(d_keys, d_next, d_buckets, numKeys, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insertKeys kernel" << std::endl;
        return -1;
    }

    // Search for keys
    numBlocks = (numSearches + blockSize - 1) / blockSize;
    searchKeys<<<numBlocks, blockSize>>>(d_keys, d_next, d_buckets, d_results, numSearches, 5678);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch searchKeys kernel" << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_results, d_results, numSearches * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy results from device to host" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_keys);
    hipFree(d_next);
    hipFree(d_buckets);
    hipFree(d_results);

    // Free host memory
    delete[] h_keys;
    delete[] h_next;
    delete[] h_buckets;
    delete[] h_results;

    std::cout << "Hash table operations completed successfully." << std::endl;
    return 0;
}