#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_GPUS 2
#define NODES_PER_GPU 1024
#define MAX_CHILDREN 4

struct BTreeNode {
    int keys[MAX_CHILDREN - 1];
    int numKeys;
    int children[MAX_CHILDREN];
    int isLeaf;
};

__device__ BTreeNode d_nodes[NODES_PER_GPU];

__global__ void initializeNodes(BTreeNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NODES_PER_GPU) {
        nodes[idx].numKeys = 0;
        nodes[idx].isLeaf = 1;
        for (int i = 0; i < MAX_CHILDREN - 1; ++i) {
            nodes[idx].keys[i] = 0;
        }
        for (int i = 0; i < MAX_CHILDREN; ++i) {
            nodes[idx].children[i] = -1;
        }
    }
}

__global__ void insertKey(BTreeNode* nodes, int key, int nodeIdx) {
    int idx = threadIdx.x;
    extern __shared__ int sharedKeys[];

    // Load keys into shared memory
    if (idx < nodes[nodeIdx].numKeys) {
        sharedKeys[idx] = nodes[nodeIdx].keys[idx];
    }
    __syncthreads();

    // Find the correct position to insert the new key
    int pos = 0;
    while (pos < nodes[nodeIdx].numKeys && sharedKeys[pos] < key) {
        pos++;
    }

    // Shift keys to the right to make space for the new key
    for (int i = nodes[nodeIdx].numKeys; i > pos; --i) {
        sharedKeys[i] = sharedKeys[i - 1];
    }
    sharedKeys[pos] = key;

    // Write back keys to global memory
    if (idx < nodes[nodeIdx].numKeys + 1) {
        nodes[nodeIdx].keys[idx] = sharedKeys[idx];
    }
    if (idx == 0) {
        nodes[nodeIdx].numKeys++;
    }
    __syncthreads();
}

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount < NUM_GPUS) {
        std::cerr << "Error: Not enough GPUs available. Required: " << NUM_GPUS << ", Available: " << deviceCount << std::endl;
        return -1;
    }

    hipError_t err;
    BTreeNode* d_nodesArray[NUM_GPUS];

    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        err = hipMalloc(&d_nodesArray[i], NODES_PER_GPU * sizeof(BTreeNode));
        if (err != hipSuccess) {
            std::cerr << "Failed to allocate memory on GPU " << i << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        err = hipMemcpy(d_nodesArray[i], d_nodes, NODES_PER_GPU * sizeof(BTreeNode), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            std::cerr << "Failed to copy memory to GPU " << i << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        int threadsPerBlock = 256;
        int blocksPerGrid = (NODES_PER_GPU + threadsPerBlock - 1) / threadsPerBlock;
        initializeNodes<<<blocksPerGrid, threadsPerBlock>>>(d_nodesArray[i]);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to initialize nodes on GPU " << i << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipDeviceSynchronize();
    }

    // Example: Insert keys into the B-tree on each GPU
    int keysToInsert[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int numKeys = sizeof(keysToInsert) / sizeof(keysToInsert[0]);

    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        for (int j = 0; j < numKeys; ++j) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (NODES_PER_GPU + threadsPerBlock - 1) / threadsPerBlock;
            insertKey<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_nodesArray[i], keysToInsert[j], 0);
            err = hipGetLastError();
            if (err != hipSuccess) {
                std::cerr << "Failed to insert key on GPU " << i << ": " << hipGetErrorString(err) << std::endl;
                return -1;
            }
            hipDeviceSynchronize();
        }
    }

    // Clean up
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        hipFree(d_nodesArray[i]);
    }

    std::cout << "B-tree operations completed successfully on " << NUM_GPUS << " GPUs." << std::endl;
    return 0;
}