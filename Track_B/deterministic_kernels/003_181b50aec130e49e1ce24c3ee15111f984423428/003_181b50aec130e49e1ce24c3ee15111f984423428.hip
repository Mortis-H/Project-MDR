
#include <hip/hip_runtime.h>
#include <iostream>

#define HD __host__ __device__

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class Data {
    public:
        int *a;
        int b;
        HD Data(int i){ a=new int(i); b=1000; }
        ~Data() { delete a; }
        HD Data(const Data& d) { *a = *(d.a); b=d.b; }
};

__global__ void increment_a(Data *d){
    *(d->a) += 1;
}

__global__ void increment_a_2(Data d){
    *(d.a) += 1;
}

__global__ void increment_b(Data *d){
    d->b += 1;
}



int main(){

    // Pointers to host and device objects
    Data *h_dat, *d_dat;
    h_dat = new Data(1);

    // Shallow copy of host to device
    gpuErrchk( hipMalloc(&d_dat, sizeof(Data)) );
    gpuErrchk( hipMemcpy(d_dat, h_dat, sizeof(Data), hipMemcpyHostToDevice) );

    // Increment b and copy back to host
    increment_b<<<1,1>>>(d_dat);
    gpuErrchk( hipMemcpy(h_dat, d_dat, sizeof(Data), hipMemcpyDeviceToHost) );
    std::cout << h_dat->b << "\n";

    //////////////////////////////////
    // So far, so good. In order to do the same with a,
    // we need to make the copy deep
    //////////////////////////////////

    // This doesn't work - cannot malloc into pointers that live themselbes on the device
    //cudaMalloc(&(d_dat->a), sizeof(int));

    // So, we create a pointer that lives on the host
    int* d_a;
    // and point it to some new device storage,
    gpuErrchk( hipMalloc(&d_a, sizeof(int)) );
    // which we copy over with the value from the host.
    gpuErrchk( hipMemcpy(d_a, h_dat->a, sizeof(int), hipMemcpyHostToDevice) ); // Put data behind it
    // Finally, we copy *the pointer itself* to the device.
    gpuErrchk( hipMemcpy(&(d_dat->a), &d_a, sizeof(void *), hipMemcpyHostToDevice) ); // copy the value of the pointer into the object on the device


    // We are now ready to increment a on device
    increment_a<<<1,1>>>(d_dat);

    // Now in order to read back the value, we can *not*
    // just use the pointer d_dat->a, since it itself is stored on the device
    // cudaMemcpy(h_dat->a, d_dat->a, sizeof(int), cudaMemcpyDeviceToHost);

    // Luckily, however, d_a should still point to the device location of interest,
    // so we can do without copying the pointer from the device first.
    gpuErrchk( hipMemcpy(h_dat->a, d_a, sizeof(int), hipMemcpyDeviceToHost) );

    // Now, *(h_dat->a) should have been incremented to 2
    std::cout << *(h_dat->a) << "\n";

    //////////////////////////////////
    // We could save a few lines here by preparing a Data object h_dat2
    // with a pointer h_dat2.a pointing to device memory and then pass
    // this object to the kernel *by value*.
    // Note that in this approach, the copy of h_dat2 on the device is 
    // lost completely; 
    // We are relying on the fact that the address where h_dat2.a is pointing
    // does not change and from there can copy the value back to the host
    ///////////////////////////////////
    //Data h_dat2 = *hdat;
    //gpuErrchk( cudaMalloc(&(h_dat2.a), sizeof(int)) );
    // Since we are overwriting the pointer here, we need to have a temporary copy of the value behind it
    // It's probably even less clever than above, where we
    // just needed to have a temporary copy of the pointer.




    return 0;
}

