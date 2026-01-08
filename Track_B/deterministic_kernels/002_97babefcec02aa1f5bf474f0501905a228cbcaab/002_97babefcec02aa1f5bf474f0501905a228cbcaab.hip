
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include<iostream>

using namespace std;

class Myclass 
{
public:
  
  
  Myclass(int a=0,int b=0)
  {_a=a;_b=b;}
  virtual __host__ __device__ void printValues() 
  {
    
    printf("a = %d, b = %d\n", _a, _b);
    
  }

private:
  int _a;
  int _b;

};

__global__ void virtualFunctions(Myclass *vf) 
{
  Myclass vf_local = Myclass(*vf);
 memcpy(vf, &vf_local, sizeof(Myclass));
  vf->printValues();
}

__global__ void callVFunc(Myclass *vf) 
{
  vf->printValues();
}

int main() {
  //CPU
  Myclass vf_host(4,5);
  

  //GPU
  Myclass *vf;
  hipMalloc(&vf, sizeof(Myclass));
  
  // CPU --> GPU
  hipMemcpy(vf, &vf_host, sizeof(Myclass), hipMemcpyHostToDevice);


  virtualFunctions<<<1, 1>>>(vf);
  hipDeviceSynchronize();
  
  callVFunc<<<1, 1>>>(vf);
  hipDeviceSynchronize();
  
  return 0;
}
