#include"stdio.h"
#include"time.h"
__global__ void gpu_1(float *da1,float *db1,float *dc1,int n)
{
	for(int i=0;i<n;i++)
	{
		//dc1[i]=db1[i]+da1[i];
		dc1[i]=db1[i]*da1[i];
		
		
	}
	
}
__global__ void gpu_2(float *da1,float *db1,float *dc1,int n)
{
	int tid=threadIdx.x;
	const int t_n=blockDim.x;
	printf("%d\n",t_n);
	while(tid<n)
	{
		//dc1[tid]=db1[tid]+da1[tid];
		dc1[tid]=db1[tid]*da1[tid];
		
		tid+=t_n;
	}
	
}
__global__ void gpu_3(float *da1,float *db1,float *dc1,int n)
{
	const int tidx=threadIdx.x;//当前线程的编号
	const int bidx=blockIdx.x;//当前block的个数
	const int t_n=gridDim.x*blockDim.x;//总block的个数乘总维度
	int tid=bidx*blockDim.x+tidx;	
	//printf("%d\n",t_n);
	while(tid<n)
	{
		//dc1[tid]=db1[tid]+da1[tid];
		dc1[tid]=db1[tid]*da1[tid];
		
		tid+=t_n;
	}
	
}
int main()
{
	const int arrsize=99999;
  	const int ARRAY_BYTES = arrsize * sizeof(float);
	float a[arrsize];
	float b[arrsize];
	float c[arrsize];
	for(int i=0;i<arrsize;i++)
	{
		b[i]=(float)(i+1);
		a[i]=(float)(i+1);
	}
clock_t start,end;
clock_t start_gpu1,end_gpu1;
start=clock();
for(int i=0;i<arrsize;i++)
	{
		//c[i]=b[i]+a[i];
		c[i]=b[i]*a[i];
		
		
	}
end=clock();
double during=(double)(end-start)/CLOCKS_PER_SEC;
printf("耗时%f秒\n",during);
printf("%f\n",c[0]);
/***********************************gpu单block单thread**************************************/
	float *da;
	float *db;
	float *dc;
	float ga1[arrsize];
	float gb1[arrsize];
	float gc1[arrsize];
	for(int i=0;i<arrsize;i++)
	{
		gb1[i]=(float)(i+1);
		ga1[i]=(float)(i+1);
	}
	cudaMalloc((void**) &da,ARRAY_BYTES);
	cudaMalloc((void**) &db,ARRAY_BYTES);
	cudaMalloc((void**) &dc,ARRAY_BYTES);
	cudaMemcpy(da,ga1,ARRAY_BYTES,cudaMemcpyHostToDevice);
	cudaMemcpy(db,gb1,ARRAY_BYTES,cudaMemcpyHostToDevice);
	start_gpu1=clock();
	//gpu_1<<<1,1>>>(da,db,dc,arrsize);
/***********************************gpu单block多thread**************************************/
	gpu_2<<<1,1024>>>(da,db,dc,arrsize);
/***********************************gpu多block多thread**************************************/
	//gpu_3<<<5000,1024>>>(da,db,dc,arrsize);
	end_gpu1=clock();
	cudaMemcpy(gc1,dc,ARRAY_BYTES,cudaMemcpyDeviceToHost);
	
double during1=(double)(end_gpu1-start_gpu1)/CLOCKS_PER_SEC;
printf("耗时%f秒\n",during1);
printf("%f\n",gc1[0]);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);


}
