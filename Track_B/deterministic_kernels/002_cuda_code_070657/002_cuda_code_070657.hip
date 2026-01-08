#include <iostream>
#include <hip/hip_runtime.h>

#define BUCKET_SIZE 10
#define TABLE_SIZE 512
#define MAX_KEY 10000

__constant__ int c_table_size = TABLE_SIZE;

__global__ void insertKernel(int* keys, int* values, int* hash_table_keys, int* hash_table_values, int num_elements) {
    extern __shared__ int shared_keys[];
    extern __shared__ int shared_values[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Load data into shared memory
    shared_keys[threadIdx.x] = keys[idx];
    shared_values[threadIdx.x] = values[idx];
    __syncthreads();

    // Insert into hash table using linear probing
    int key = shared_keys[threadIdx.x];
    int value = shared_values[threadIdx.x];
    int hash = key % c_table_size;
    int step = 1;

    while (true) {
        int slot = hash % c_table_size;
        int old_key = atomicCAS(&hash_table_keys[slot], 0, key);
        if (old_key == 0) {
            hash_table_values[slot] = value;
            break;
        } else if (old_key == key) {
            // Key already exists, update value
            atomicExch(&hash_table_values[slot], value);
            break;
        } else {
            // Collision, try next slot
            hash += step;
            step++;
        }
    }
}

__global__ void searchKernel(int* search_keys, int* search_results, int* hash_table_keys, int* hash_table_values, int num_searches) {
    extern __shared__ int shared_keys[];
    extern __shared__ int shared_results[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_searches) return;

    // Load data into shared memory
    shared_keys[threadIdx.x] = search_keys[idx];
    __syncthreads();

    // Search in hash table using linear probing
    int key = shared_keys[threadIdx.x];
    int hash = key % c_table_size;
    int step = 1;

    while (true) {
        int slot = hash % c_table_size;
        int current_key = hash_table_keys[slot];
        if (current_key == key) {
            shared_results[threadIdx.x] = hash_table_values[slot];
            break;
        } else if (current_key == 0) {
            shared_results[threadIdx.x] = -1; // Key not found
            break;
        } else {
            // Collision, try next slot
            hash += step;
            step++;
        }
    }
    __syncthreads();

    // Write result back to global memory
    search_results[idx] = shared_results[threadIdx.x];
}

int main() {
    int num_elements = 1024;
    int num_searches = 128;

    int* h_keys = new int[num_elements];
    int* h_values = new int[num_elements];
    int* h_search_keys = new int[num_searches];
    int* h_search_results = new int[num_searches];

    // Initialize keys and values
    for (int i = 0; i < num_elements; i++) {
        h_keys[i] = rand() % MAX_KEY;
        h_values[i] = rand() % MAX_KEY;
    }

    // Initialize search keys
    for (int i = 0; i < num_searches; i++) {
        h_search_keys[i] = h_keys[rand() % num_elements];
    }

    int* d_keys, *d_values, *d_search_keys, *d_search_results;
    int* d_hash_table_keys, *d_hash_table_values;

    hipError_t err;

    err = hipMalloc(&d_keys, num_elements * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_keys: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_values, num_elements * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_values: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_search_keys, num_searches * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_search_keys: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_search_results, num_searches * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_search_results: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_hash_table_keys, TABLE_SIZE * BUCKET_SIZE * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_hash_table_keys: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_hash_table_values, TABLE_SIZE * BUCKET_SIZE * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_hash_table_values: " << hipGetErrorString(err) << std::endl; return -1; }

    err = hipMemcpy(d_keys, h_keys, num_elements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying d_keys: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_values, h_values, num_elements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying d_values: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_search_keys, h_search_keys, num_searches * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying d_search_keys: " << hipGetErrorString(err) << std::endl; return -1; }

    // Initialize hash table keys to 0
    err = hipMemset(d_hash_table_keys, 0, TABLE_SIZE * BUCKET_SIZE * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error initializing d_hash_table_keys: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemset(d_hash_table_values, 0, TABLE_SIZE * BUCKET_SIZE * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error initializing d_hash_table_values: " << hipGetErrorString(err) << std::endl; return -1; }

    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    insertKernel<<<numBlocks, blockSize, 2 * blockSize * sizeof(int)>>>(d_keys, d_values, d_hash_table_keys, d_hash_table_values, num_elements);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error in insertKernel: " << hipGetErrorString(err) << std::endl; return -1; }
    hipDeviceSynchronize();

    numBlocks = (num_searches + blockSize - 1) / blockSize;
    searchKernel<<<numBlocks, blockSize, 2 * blockSize * sizeof(int)>>>(d_search_keys, d_search_results, d_hash_table_keys, d_hash_table_values, num_searches);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error in searchKernel: " << hipGetErrorString(err) << std::endl; return -1; }
    hipDeviceSynchronize();

    err = hipMemcpy(h_search_results, d_search_results, num_searches * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "Error copying d_search_results: " << hipGetErrorString(err) << std::endl; return -1; }

    // Output some results for verification
    for (int i = 0; i < 10; i++) {
        std::cout << "Search key: " << h_search_keys[i] << ", Found value: " << h_search_results[i] << std::endl;
    }

    hipFree(d_keys);
    hipFree(d_values);
    hipFree(d_search_keys);
    hipFree(d_search_results);
    hipFree(d_hash_table_keys);
    hipFree(d_hash_table_values);

    delete[] h_keys;
    delete[] h_values;
    delete[] h_search_keys;
    delete[] h_search_results;

    return 0;
}