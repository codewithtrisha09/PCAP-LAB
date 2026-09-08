#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        printf("CUDA Error at %s:%d -> %s\n",                \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

void que1(){
printf("\nCUDA DEVICE PROPERTIES\n");
int deviceCount=0;
CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
printf("Number of CUDA capable devices:%d",deviceCount);
if (deviceCount==0){
printf("No CUDA capable device found");
return;
}
for (int i = 0; i < deviceCount; i++) {
cudaDeviceProp prop;
CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
printf("\nDevice %d\n", i);
printf("Device Name: %s\n", prop.name);
printf("Streaming Multiprocessors: %d\n",prop.multiProcessorCount);
printf("Max Threads Per Block: %d\n",prop.maxThreadsPerBlock);
printf("Max Thread Dimensions: (%d, %d, %d)\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
printf("Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
printf("Clock Frequency: %d kHz\n",prop.clockRate);
printf("Compute Capability: %d.%d\n",prop.major, prop.minor);
}
}

int main(){
que1();
return 0;
}



