#include <iostream>
#include <hip/hip_runtime.h>


// Define the maximum number of elements in the trie
#define MAX_ELEMENTS 32

// Trie node structure
struct TrieNode {
    bool isEndOfWord;
    TrieNode* children[2]; // Binary trie for simplicity
};

// Kernel to initialize the trie nodes
__global__ void initTrieNodes(TrieNode* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        nodes[idx].isEndOfWord = false;
        nodes[idx].children[0] = nullptr;
        nodes[idx].children[1] = nullptr;
    }
}

// Kernel to insert a binary number into the trie
__global__ void insertIntoTrie(TrieNode* root, int* numbers, int numElements, int numBits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        TrieNode* current = root;
        for (int i = numBits - 1; i >= 0; --i) {
            int bit = (numbers[idx] >> i) & 1;
            if (current->children[bit] == nullptr) {
                // This is not thread-safe and assumes single-threaded execution for simplicity
                current->children[bit] = &root[numElements + (idx * numBits) + i];
            }
            current = current->children[bit];
        }
        current->isEndOfWord = true;
    }
}

// Kernel to search for a binary number in the trie
__global__ void searchInTrie(TrieNode* root, int* numbers, bool* results, int numElements, int numBits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        TrieNode* current = root;
        for (int i = numBits - 1; i >= 0; --i) {
            int bit = (numbers[idx] >> i) & 1;
            if (current->children[bit] == nullptr) {
                results[idx] = false;
                return;
            }
            current = current->children[bit];
        }
        results[idx] = current->isEndOfWord;
    }
}

int main() {
    const int numElements = MAX_ELEMENTS;
    const int numBits = 5; // Example: 5-bit numbers

    // Allocate host memory
    int* h_numbers = new int[numElements];
    bool* h_results = new bool[numElements];

    // Initialize numbers to insert into the trie
    for (int i = 0; i < numElements; ++i) {
        h_numbers[i] = i; // Insert numbers 0 to 31
        h_results[i] = false;
    }

    // Allocate device memory
    TrieNode* d_root;
    int* d_numbers;
    bool* d_results;
    size_t nodeSize = sizeof(TrieNode) * (numElements + numElements * numBits); // Extra space for child nodes
    hipMalloc(&d_root, nodeSize);
    hipMalloc(&d_numbers, numElements * sizeof(int));
    hipMalloc(&d_results, numElements * sizeof(bool));

    // Initialize trie nodes on the device
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    initTrieNodes<<<numBlocks, blockSize>>>(d_root, numElements + numElements * numBits);
    hipDeviceSynchronize();

    // Copy data to the device
    hipMemcpy(d_numbers, h_numbers, numElements * sizeof(int), hipMemcpyHostToDevice);

    // Insert numbers into the trie
    insertIntoTrie<<<numBlocks, blockSize>>>(d_root, d_numbers, numElements, numBits);
    hipDeviceSynchronize();

    // Search for numbers in the trie
    searchInTrie<<<numBlocks, blockSize>>>(d_root, d_numbers, d_results, numElements, numBits);
    hipDeviceSynchronize();

    // Copy results back to the host
    hipMemcpy(h_results, d_results, numElements * sizeof(bool), hipMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < numElements; ++i) {
        std::cout << "Number " << h_numbers[i] << " is in the trie: " << (h_results[i] ? "true" : "false") << std::endl;
    }

    // Free device memory
    hipFree(d_root);
    hipFree(d_numbers);
    hipFree(d_results);

    // Free host memory
    delete[] h_numbers;
    delete[] h_results;

    return 0;
}