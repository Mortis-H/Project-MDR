#include <iostream>
#include <hip/hip_runtime.h>

#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 16384
#define MAX_LEVEL 16
#define NUM_GPUS 2

struct Node {
    int key;
    int value;
    Node* forward[MAX_LEVEL];
};

__global__ void initializeNodes(Node* nodes, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        nodes[idx].key = idx;
        nodes[idx].value = idx * 2; // Example value
        for (int i = 0; i < MAX_LEVEL; ++i) {
            nodes[idx].forward[i] = nullptr;
        }
    }
}

__device__ int randomLevel(hiprandState* localState) {
    int lvl = 1;
    while (hiprand_uniform(localState) < 0.5 && lvl < MAX_LEVEL) {
        lvl++;
    }
    return lvl;
}

__global__ void insertNodes(Node* nodes, Node** head, int numElements, hiprandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        hiprandState localState = states[idx];
        int lvl = randomLevel(&localState);
        states[idx] = localState;

        Node* newNode = &nodes[idx];
        Node* update[MAX_LEVEL];
        Node* current = *head;

        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            while (current->forward[i] != nullptr && current->forward[i]->key < newNode->key) {
                current = current->forward[i];
            }
            update[i] = current;
        }

        for (int i = 0; i < lvl; ++i) {
            newNode->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = newNode;
        }
    }
}

__global__ void setupCurandStates(hiprandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    Node* h_nodes[NUM_GPUS];
    Node** d_heads[NUM_GPUS];
    hiprandState* d_states[NUM_GPUS];

    int elementsPerGPU = NUM_ELEMENTS / NUM_GPUS;
    int threadsPerBlock = 256;
    int blocksPerGrid = (elementsPerGPU + threadsPerBlock - 1) / threadsPerBlock;

    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);

        hipMalloc(&h_nodes[gpu], elementsPerGPU * sizeof(Node));
        hipMalloc(&d_heads[gpu], sizeof(Node*));
        hipMalloc(&d_states[gpu], elementsPerGPU * sizeof(hiprandState));

        Node* d_head;
        hipMalloc(&d_head, sizeof(Node));
        hipMemset(d_head, 0, sizeof(Node));
        hipMemcpy(d_heads[gpu], &d_head, sizeof(Node*), hipMemcpyHostToDevice);

        setupCurandStates<<<blocksPerGrid, threadsPerBlock>>>(d_states[gpu], time(0));
        hipDeviceSynchronize();

        initializeNodes<<<blocksPerGrid, threadsPerBlock>>>(h_nodes[gpu], elementsPerGPU);
        hipDeviceSynchronize();

        insertNodes<<<blocksPerGrid, threadsPerBlock>>>(h_nodes[gpu], d_heads[gpu], elementsPerGPU, d_states[gpu]);
        hipDeviceSynchronize();

        hipFree(d_head);
    }

    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(h_nodes[gpu]);
        hipFree(d_heads[gpu]);
        hipFree(d_states[gpu]);
    }

    hipDeviceReset();
    return 0;
}