// skip_list_cuda.cu
// CUDA implementation of skip list operations optimized for Turing architecture

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 8192
#define MAX_LEVEL 16

// Node structure for skip list
struct Node {
    int value;
    int level;
    Node* next[MAX_LEVEL];
};

// Global device pointers
__device__ Node* d_head;

// Function to initialize the skip list head
__global__ void initSkipList() {
    extern __shared__ Node sharedHead[];
    Node* head = sharedHead;
    head->value = -1;
    head->level = MAX_LEVEL - 1;
    for (int i = 0; i < MAX_LEVEL; i++) {
        head->next[i] = nullptr;
    }
    d_head = head;
}

// Random level generator
__device__ int randomLevel(hiprandState* localState) {
    int lvl = 1;
    while (hiprand_uniform(localState) < 0.5 && lvl < MAX_LEVEL) {
        lvl++;
    }
    return lvl;
}

// Insert operation in skip list
__global__ void insertSkipList(int* values, int numValues) {
    extern __shared__ Node sharedNodes[];
    hiprandState localState;
    hiprand_init(threadIdx.x + blockIdx.x * blockDim.x, 0, 0, &localState);

    for (int i = threadIdx.x; i < numValues; i += blockDim.x * gridDim.x) {
        int value = values[i];
        Node* update[MAX_LEVEL];
        Node* x = d_head;

        // Find the position to insert
        for (int i = x->level; i >= 0; i--) {
            while (x->next[i] != nullptr && x->next[i]->value < value) {
                x = x->next[i];
            }
            update[i] = x;
        }
        x = x->next[0];

        // If value is not present in skip list
        if (x == nullptr || x->value != value) {
            int lvl = randomLevel(&localState);
            if (lvl > x->level) {
                for (int i = x->level + 1; i <= lvl; i++) {
                    update[i] = d_head;
                }
                x->level = lvl;
            }

            // Create new node
            Node* newNode = &sharedNodes[i];
            newNode->value = value;
            newNode->level = lvl;

            // Insert node by rearranging pointers
            for (int i = 0; i <= lvl; i++) {
                newNode->next[i] = update[i]->next[i];
                update[i]->next[i] = newNode;
            }
        }
    }
}

// Function to print the skip list (for demonstration purposes)
__global__ void printSkipList() {
    Node* x = d_head;
    while (x != nullptr) {
        printf("%d ", x->value);
        x = x->next[0];
    }
    printf("\n");
}

int main() {
    int numValues = NUM_ELEMENTS;
    int* h_values = new int[numValues];
    for (int i = 0; i < numValues; i++) {
        h_values[i] = i + 1; // Simple ascending order for demonstration
    }

    int* d_values;
    hipMalloc(&d_values, numValues * sizeof(int));
    hipMemcpy(d_values, h_values, numValues * sizeof(int), hipMemcpyHostToDevice);

    // Allocate memory for head node
    size_t sharedMemSize = sizeof(Node);
    initSkipList<<<1, 1, sharedMemSize>>>();
    hipDeviceSynchronize();

    // Insert values into skip list
    insertSkipList<<<128, 256, numValues * sizeof(Node)>>>(d_values, numValues);
    hipDeviceSynchronize();

    // Print skip list
    printSkipList<<<1, 1>>>();
    hipDeviceSynchronize();

    // Free memory
    hipFree(d_values);
    delete[] h_values;

    return 0;
}