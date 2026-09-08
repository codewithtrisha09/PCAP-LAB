#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
void sineKernel(float *input,float *output,int N){
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i<N){
output[i]=sinf(input[i]);
}
}

void que4(){
printf("\nSINE OF ANGLES\n");

int N=10;

size_t size=N*sizeof(float);

float *h_input=(float*)malloc(size);
float *h_output=(float*)malloc(size);

float *d_input;
float *d_output;

h_input[0]=0.0f;
h_input[1]=0.523599f;
h_input[2]=0.785398f;
h_input[3]=1.047198f;
h_input[4]=1.570796f;
h_input[5]=2.094395f;
h_input[6]=2.356194f;
h_input[7]=3.141593f;
h_input[8]=4.712389f;
h_input[9]=6.283185f;

CUDA_CHECK(cudaMalloc((void**)&d_input,size));
CUDA_CHECK(cudaMalloc((void**)&d_output,size));

CUDA_CHECK(cudaMemcpy(d_input,h_input,size,cudaMemcpyHostToDevice));

int threadsPerBlock=256;
int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;

printf("Threads per block: %d\n",threadsPerBlock);
printf("Number of blocks: %d\n",blocksPerGrid);

sineKernel<<<blocksPerGrid,threadsPerBlock>>>(d_input,d_output,N);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

CUDA_CHECK(cudaMemcpy(h_output,d_output,size,cudaMemcpyDeviceToHost));

printf("\nAngle(radians)    sin(angle)\n");


for(int i=0;i<N;i++){
printf("%12.6f    %8.6f\n",h_input[i],h_output[i]);
}

CUDA_CHECK(cudaFree(d_input));
CUDA_CHECK(cudaFree(d_output));

free(h_input);
free(h_output);
}

int main(){
que4();
return 0;
}

