#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>

__global__ void modular(int *a, int *b, int *c){
    // for the small case, we only need thread, no block and grid
    int i = threadIdx.x;
    c[i] = a[i] % b[i];
    // printf is not allowed in kernel function
    // printf("%d", c[i])
}

__global__ void exponentiation(int *a, int *b, int *d){
    // for the small case, we only need thread, no block and grid
    int i = threadIdx.x;
    d[i] = pow(a[i], b[i]);
}

int main(){
    int a[3] = {1, 2, 5}, b[3] = {2, 4, 6};
    int c[3], d[3], i;
    int c_check[3],  d_exponentiation[3];
    int *A_gpu, *B_gpu, *C_gpu, *D_gpu; // pointers
    int size = 5 * sizeof(int);

    // allocate memory for A, B, and C
    hipMalloc((void **)&A_gpu, size);
    hipMalloc((void **)&B_gpu, size);
    hipMalloc((void **)&C_gpu, size);
    hipMalloc((void **)&D_gpu, size);

    // copy the memory
    hipMemcpy(A_gpu, a, size, hipMemcpyHostToDevice);
    hipMemcpy(B_gpu, b, size, hipMemcpyHostToDevice);

    modular<<<1, 3>>>(A_gpu, B_gpu, C_gpu);
    exponentiation<<<1, 3>>>(A_gpu, B_gpu, D_gpu);

    // copy memory from device to host 
    // since the result is stored in C_gpu
    // we just need this memory
    hipMemcpy(c, C_gpu, size, hipMemcpyDeviceToHost);
    hipMemcpy(d, D_gpu, size, hipMemcpyDeviceToHost);

    hipFree(A_gpu);
    hipFree(B_gpu);
    hipFree(C_gpu);
    hipFree(D_gpu);
    
    for (i = 0; i < 3; i++){
        c_check[i] = a[i] % b[i];
        printf("The GPU Version %d, The CPU Version is %d\n", c[i], c_check[i]);
    }  
    
//TODO: why the CPU version in exponentiation is wrong ? WHY ?
    for (i = 0; i < 3; i++){
        d_exponentiation[i] = pow(a[i], b[i]);
        printf("The GPU Version %d, The CPU Version is %d\n", d[i], d_exponentiation[i]);
    }

    return 0;
}

