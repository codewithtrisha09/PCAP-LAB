#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
cudaError_t err=call; \
if(err!=cudaSuccess){ \
printf("CUDA Error at %s:%d -> %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while(0)

__global__
void reverseArrayKernel(int *input,int *output,int N){
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i<N){
int reverseIndex=N-1-i;
output[reverseIndex]=input[i];
}
}

void additionalExercise2(){
printf("\nREVERSE ARRAY\n");

int N=100;

size_t size=N*sizeof(int);

int *h_input=(int*)malloc(size);
int *h_output=(int*)malloc(size);

int *d_input;
int *d_output;

srand(1234);

for(int i=0;i<N;i++){
h_input[i]=rand()%100;
}

printf("\nOriginal array:\n");

for(int i=0;i<20;i++){
printf("%d ",h_input[i]);
}

printf("\n");

CUDA_CHECK(cudaMalloc((void**)&d_input,size));
CUDA_CHECK(cudaMalloc((void**)&d_output,size));

CUDA_CHECK(cudaMemcpy(d_input,h_input,size,cudaMemcpyHostToDevice));

int threadsPerBlock=32;
int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;

printf("\nArray size        : %d",N);
printf("\nThreads per block : %d",threadsPerBlock);
printf("\nNumber of blocks  : %d\n",blocksPerGrid);

reverseArrayKernel<<<blocksPerGrid,threadsPerBlock>>>(d_input,d_output,N);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

CUDA_CHECK(cudaMemcpy(h_output,d_output,size,cudaMemcpyDeviceToHost));

printf("\nReversed array:\n");

for(int i=0;i<20;i++){
printf("%d ",h_output[i]);
}

printf("\n");

CUDA_CHECK(cudaFree(d_input));
CUDA_CHECK(cudaFree(d_output));

free(h_input);
free(h_output);
}

int main(){
additionalExercise2();
return 0;
}

