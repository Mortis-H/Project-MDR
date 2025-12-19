#include <hip/hip_runtime.h>

// ============================================================================
// Kernel 1: Vector Addition
// ============================================================================
// 功能：c[i] = a[i] + b[i]
extern "C" __global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// Kernel 2: Vector Multiplication
// ============================================================================
// 功能：c[i] = a[i] * b[i]
extern "C" __global__ void vectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// ============================================================================
// Kernel 3: Vector Subtraction
// ============================================================================
// 功能：c[i] = a[i] - b[i]
extern "C" __global__ void vectorSub(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// ============================================================================
// Kernel 4: Scalar Operations (使用不同數量的寄存器)
// ============================================================================
// 功能：複雜的純量運算，使用更多寄存器
extern "C" __global__ void scalarOps(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int a = idx;
        int b = idx + 1;
        int c = a & b;      // AND
        int d = a | b;      // OR
        int e = a ^ b;      // XOR
        int f = a >> 2;     // Shift right
        int g = b << 1;     // Shift left
        int result = c + d - e + f * g;
        output[idx] = result;
    }
}

