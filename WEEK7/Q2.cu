#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
printf("CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while(0)
__global__
void vectorAddKernel(float*A,float*B,float*C,int N){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i<N){
C[i]=A[i]+B[i];
}
}
__global__
void dotProductKernel(float*A,float*B,float*result,int N){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i<N){
atomicAdd(result,A[i]*B[i]);
}
}
void que2(){
int N=256;
size_t size=N*sizeof(float);
float *h_A=(float*)malloc(size);
float *h_B=(float*)malloc(size);
float *h_C=(float*)malloc(size);
float *d_A;
float *d_B;
float *d_C;
float *d_result;
for(int i=0;i<N;i++){
h_A[i]=(float)i;
h_B[i]=(float)(2*i);
}
CUDA_CHECK(cudaMalloc((void**)&d_A,size));
CUDA_CHECK(cudaMalloc((void**)&d_B,size));
CUDA_CHECK(cudaMalloc((void**)&d_C,size));
CUDA_CHECK(cudaMalloc((void**)&d_result,sizeof(float)));
CUDA_CHECK(cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice));
printf("\nVector Addition\n");
vectorAddKernel<<<1,N>>>(d_A,d_B,d_C,N);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
CUDA_CHECK(cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost));
printf("First 10 elements of A + B:\n");
for(int i=0;i<10;i++){
printf("C[%d] = %.2f\n",i,h_C[i]);
}
printf("\nDot Product\n");
float zero=0.0f;
CUDA_CHECK(cudaMemcpy(d_result,&zero,sizeof(float),cudaMemcpyHostToDevice));
dotProductKernel<<<1,N>>>(d_A,d_B,d_result,N);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
float dotProduct;
CUDA_CHECK(cudaMemcpy(&dotProduct,d_result,sizeof(float),cudaMemcpyDeviceToHost));
printf("Dot Product = %.2f\n",dotProduct);
CUDA_CHECK(cudaFree(d_A));
CUDA_CHECK(cudaFree(d_B));
CUDA_CHECK(cudaFree(d_C));
CUDA_CHECK(cudaFree(d_result));
free(h_A);
free(h_B);
free(h_C);}
int main(){
que2();
return 0;
}

