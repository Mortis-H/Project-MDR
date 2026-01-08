// binary_tree.cu
// CUDA implementation for binary tree operations with 16384 elements
// Target architecture: Hopper

#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_ELEMENTS 16384
#define MAX_TREE_HEIGHT 15  // log2(16384)

__global__ void build_binary_tree(int *data, int *tree, int n) {
    // Each thread processes one node in the binary tree
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Initialize leaf nodes with data
        tree[n + idx] = data[idx];
    } else {
        // Initialize non-leaf nodes to 0
        tree[idx] = 0;
    }
    __syncthreads();

    // Build the tree bottom-up
    for (int s = n / 2; s > 0; s >>= 1) {
        if (idx < s) {
            tree[idx] = tree[2 * idx] + tree[2 * idx + 1];
        }
        __syncthreads();
    }
}

__global__ void update_binary_tree(int *tree, int n, int index, int new_val) {
    // Update the tree with new value at a specific index
    int pos = n + index;
    tree[pos] = new_val;
    while (pos > 1) {
        int parent = pos / 2;
        int left = parent * 2;
        int right = left + 1;
        tree[parent] = tree[left] + tree[right];
        pos = parent;
    }
}

__global__ void query_binary_tree(int *tree, int *result, int n, int left, int right) {
    // Perform a range sum query on the binary tree
    left += n;
    right += n;
    int sum = 0;
    while (left <= right) {
        if (left & 1) {
            sum += tree[left];
            left++;
        }
        if (!(right & 1)) {
            sum += tree[right];
            right--;
        }
        left >>= 1;
        right >>= 1;
    }
    *result = sum;
}

int main() {
    int *h_data, *h_tree, *h_result;
    int *d_data, *d_tree, *d_result;
    int n = NUM_ELEMENTS;
    int tree_size = 2 * n - 1;

    // Allocate host memory
    h_data = (int *)malloc(n * sizeof(int));
    h_tree = (int *)malloc(tree_size * sizeof(int));
    h_result = (int *)malloc(sizeof(int));

    // Initialize data
    for (int i = 0; i < n; i++) {
        h_data[i] = 1;  // Example initialization
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_data, n * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_tree, tree_size * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_result, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_data, h_data, n * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Build the binary tree
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    build_binary_tree<<<numBlocks, blockSize>>>(d_data, d_tree, n);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Perform an update
    int update_index = 1024;
    int new_value = 5;
    update_binary_tree<<<1, 1>>>(d_tree, n, update_index, new_value);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Perform a query
    int query_left = 0;
    int query_right = 1024;
    query_binary_tree<<<1, 1>>>(d_tree, d_result, n, query_left, query_right);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Copy result back to host
    err = hipMemcpy(h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Output the result
    std::cout << "Sum of elements from index " << query_left << " to " << query_right << " is: " << *h_result << std::endl;

    // Free device memory
    hipFree(d_data);
    hipFree(d_tree);
    hipFree(d_result);

    // Free host memory
    free(h_data);
    free(h_tree);
    free(h_result);

    return 0;
}