#include <iostream>
#include <string>
#include <cassert>//断言判断
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


using namespace cv;
using namespace std;
Mat imageRGBA;
Mat inmageGray;

uchar4   *d_rgbaImage__;//四位uchar,GPU端rgb图地址
unsigned char *d_greyImage__;//gpu端灰度图地址

size_t numRows() {return imageRGBA.rows;}
size_t numCols() {return imageRGBA.cols;}//得到宽高

void preprocess(uchar4 **inputImage,unsigned char **greyImage,uchar4 **d_rgbaImage,unsigned char **d_greyImage,
		const string &filename){


	 cv::Mat image;
         image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);//读图

	cvtColor(image,imageRGBA,CV_BGR2RGBA);
	 inmageGray.create(image.rows,image.cols,CV_8UC1);//cpu端灰度图
	*inputImage=(uchar4 *)imageRGBA.ptr<unsigned char>(0);//第0行首地址,指针形式为多位	
	*greyImage=inmageGray.ptr<unsigned char>(0);//单通道第0行首地址，图像的首地址,在子函数中获得图像首地址
	const size_t numpixels=numRows()*numCols();
	cudaMalloc(d_rgbaImage,sizeof(uchar4) * numpixels);//GPU的rgb端地址，声明显存
	cudaMalloc(d_greyImage,sizeof(unsigned char) * numpixels);//Gpu端的灰度图地址，声明对应长度的显存，cpu地址变GPU地址
	cudaMemset(*d_greyImage,0,numpixels * sizeof(unsigned char));//初始化
	cudaMemcpy(*d_rgbaImage,*inputImage,sizeof(uchar4) * numpixels,cudaMemcpyHostToDevice);//把cpu数据传输到GPU
        d_rgbaImage__ = *d_rgbaImage;//GPU地址
	d_greyImage__ = *d_greyImage;

	}


__global__ void rgba_to_greyscale(const uchar4* const rgbaImage,unsigned char* const greyImage,int numRows,int numCols){

		int threadId=blockIdx.x * blockDim.x*blockDim.y+threadIdx.y * blockDim.x+threadIdx.x;
			if(threadId < numRows * numCols){
				const unsigned char R = rgbaImage[threadId].x;
				const unsigned char G = rgbaImage[threadId].y;
				const unsigned char B = rgbaImage[threadId].z;				

				greyImage[threadId]=  0.299f * R + 0.587f * G + 0.114f * B;


				}


}

void postprocess(const string& output_file,unsigned char* data_ptr){
		Mat output(numRows(),numCols(),CV_8UC1,(void*)data_ptr);
		imwrite(output_file.c_str(),output);

}
void cleanup(){
	cudaFree(d_rgbaImage__);//全局释放

}
int main(int argc,char* argv[])
{
	string input_file =argv[1];
	string output_file=argv[2];
	uchar4 *h_rgbaImage,*d_rgbaImage;
	unsigned char *h_greyImage,*d_greyImage;//灰度图的指针变量为空，灰度图指针变量的地址假设0x99，将灰度图指针变量的地址传入preprocess，greyImage成为
//cpu端灰度图指针变量的地址，*greyImage改为初始化图后的确定地址，
	preprocess(&h_rgbaImage,&h_greyImage,&d_rgbaImage,&d_greyImage,input_file);
	int thread=16;
	int grid =(numRows()*numCols()+thread-1)/(thread*thread);
	const dim3 blocksize(thread,thread);
	const dim3 gridsize(grid);
	rgba_to_greyscale<<<gridsize,blocksize>>>(d_rgbaImage,d_greyImage,numRows(),numCols());
	cudaDeviceSynchronize();
	size_t numPixels = numRows()*numCols();
	cudaMemcpy(h_greyImage,d_greyImage,sizeof(unsigned char)* numPixels,cudaMemcpyDeviceToHost);
	postprocess(output_file,h_greyImage);
	cleanup();



}


