#include <torch/extension.h>
//#include <torch/serialize/tensor.h>
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 256
#define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

#ifdef __cplusplus
    extern "C" {
#endif



__global__ void nlf_down_forward(const int n, const float* filters, const int channel, const int height,const int width,const int wsize, float* top_data) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step = height * width;
    
    int base = index * step; //up->down
    int fbase = index/channel*wsize*step;

    for(int row = 0; row < height; row ++){
	    for(int col = 0; col < width; col++){
			float temp = 0;
			int r = row;
			int c = col;
			int shift = 0 * step + row * width + col;
			temp += top_data[base + r*width + c] * filters[fbase + shift];
			
			r = row - 1;
			c = col;
			shift = 1 * step + row * width + col;
			if(r >= 0)
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];
			
			r = row - 1;
			c = col - 1;
			shift = 2 * step + row * width + col;
			if(r >= 0 && c >= 0)
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];			
			
			r = row - 1;
			c = col + 1;
			shift = 3 * step + row * width + col;
			if(r >= 0 && c < width)
				temp += top_data[base + r*width+c] * filters[fbase + shift];
			else
				temp += top_data[base + row*width+col] * filters[fbase + shift];

			r = row;
			c = col - 1;
			shift = 4 * step + row * width + col;
			if(c >= 0)
				temp += top_data[base + r*width+c] * filters[fbase + shift];
			else
				temp += top_data[base + row*width+col] * filters[fbase + shift];

			top_data[base + row*width + col] = temp;
		}
    }

}

__global__ void nlf_up_forward(const int n, const float* filters, const int channel, const int height,const int width,const int wsize, float* top_data) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step = height * width;
    
    int base = index*step; //down->up
    int fbase = index / channel * wsize * step;

    for(int row = height - 1; row >= 0; row --){
	    for(int col = width-1; col >=0; col--){   
			float temp = 0;
			int r = row;
			int c = col;
			int shift = 0 * step + row * width + col;
			temp += top_data[base + r*width+c]*filters[fbase + shift];
			
			r = row + 1;
			c = col;
			shift = 1 * step + row * width + col;
			if(r < height) //changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];
			
			r = row + 1;
			c = col - 1;
			shift = 2 * step + row * width + col;
			if(r < height && c >=0)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];			
			
			r = row + 1;
			c = col + 1;
			shift = 3 * step + row * width + col;
			if(r < height && c < width)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];

			r = row;
			c = col + 1;
			shift = 4 * step + row * width + col;
			if(c < width)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];
			
			top_data[base + row*width + col]=temp;
		}
    }

}


__global__ void nlf_right_forward(const int n, const float* filters, const int channel, const int height,const int width, const int wsize, float* top_data) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step=height * width;
    
    int base = index*step; //left->right
    int fbase = index / channel * wsize * step;
	for(int col = 0; col < width; col++){   
		for(int row = 0; row < height; row ++){ //changed
			float temp = 0;
			int r = row;
			int c = col;
			int shift = 0 * step + row * width + col;
			temp += top_data[base + r*width+c]*filters[fbase + shift];
			
			r = row;
			c = col - 1; //changed
			shift = 1 * step + row * width + col;
			if(c >= 0) //changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];
			
			r = row - 1; //changed
			c = col - 1;
			shift = 2 * step + row * width + col;
			if(c >= 0 && r >=0)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];			
			
			r = row + 1;
			c = col - 1; //changed
			shift = 3 * step + row * width + col;
			if(c >= 0 && r < height)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];

			r = row - 1;
			c = col; //changed
			shift = 4 * step + row * width + col;
			if(r >= 0)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];
			
			top_data[base + row*width + col] = temp;
		}
    }

}

__global__ void nlf_left_forward(const int n, const float* filters, const int channel, const int height, const int width, const int wsize, float* top_data) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step = height * width;
    
    int base = index * step; //right->left
    int fbase = index / channel * wsize * step;
	for(int col = width - 1; col >= 0; col --){   
		for(int row = height-1; row >= 0; row --){ //changed

			float temp = 0;
			int r = row;
			int c = col;
			int shift = 0 * step + row * width + col;
			temp += top_data[base + r*width+c] * filters[fbase + shift];
			
			r = row;
			c = col + 1; //changed
			shift = 1 * step + row * width + col;
			if(c < width) //changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width+col]*filters[fbase + shift];
			
			r = row - 1; //changed
			c = col + 1;
			shift = 2 * step + row * width + col;
			if(c < width && r >= 0)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width + col]*filters[fbase + shift];			
			
			r = row + 1;
			c = col + 1; //changed
			shift = 3 * step + row * width + col;
			if(c < width && r < height)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width + col]*filters[fbase + shift];

			r = row + 1;
			c = col; //changed
			shift = 4 * step + row * width + col;
			if(r < height)//changed
				temp += top_data[base + r*width+c]*filters[fbase + shift];
			else
				temp += top_data[base + row*width + col]*filters[fbase + shift];
			
			top_data[base + row*width + col] = temp;
		}
    }

}



__global__ void nlf_down_backward(const int n, const float* filters, float* top_diff, const int channel, const int height, const int width, const int wsize,float* bottom_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    int step = height * width;
    int base = index * step; //up->down
    int fbase = index/channel * step * wsize;

	for(int row = height - 1; row >= 0; row --){
	    for(int col = width - 1; col >= 0; col --){   
			float temp = top_diff[base + row * width + col];

//			int r = row;
//			int c = col;
//			int shift = 0 * step + row * width + col;
//			temp += top_data[base + r*width+c]*filters[fbase + shift];

			int r = row + 1;
			int c = col;
			int shift = 1 * step + r * width + c;
			if(r < height)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];
			
			r = row + 1;
			c = col + 1;
			shift = 2 * step + r * width + c;
			if(r < height && c < width)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			r = row + 1;
			c = col - 1;
			shift = 3 * step + r * width + c;
			if(r < height && c >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			r = row;
			c = col + 1;
			shift = 4 * step + r * width + c;
			if(c < width)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			shift = row * width + col;
		    top_diff[base + shift] = temp;
            bottom_diff[base + shift] += temp * filters[fbase + shift];	

		}
    }
	

    for(int col=0; col<width; col++){
		int location = base + col;
		int shift = fbase + col;
		bottom_diff[location] += top_diff[location] * filters[shift + step];
	}
	for (int row = 0; row < height; row++)
    {
      int location = base + row * width;
      int shift = fbase + row * width;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
      bottom_diff[location] += top_diff[location] * filters[shift + 4 * step];
	  
      location += width-1;
	  shift += width-1;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
    }

}

__global__ void nlf_up_backward(const int n, const float* filters, float* top_diff, const int channel, const int height, const int width, const int wsize,float* bottom_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
	
    int step = height * width;
    int base = index * step; //up->down
    int fbase = index/channel * step * wsize;
	
	for(int row = 0; row < height; row ++){
	    for(int col =0; col< width; col++){   
			float temp = top_diff[base + row * width + col];
//			int r = row;
//			int c = col;
//			int shift = 0 * step + row * width + col;
//			temp += top_data[base + r*width+c]*filters[fbase + shift];
			
			int r = row - 1;
			int c = col;
			int shift = 1 * step + r * width + c;
			if(r >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];
			
			r = row - 1;
			c = col + 1;
			shift = 2 * step + r * width + c;
			if(r >= 0 && c < width)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			
			r = row - 1;
			c = col - 1;
			shift = 3 * step + r * width + c;
			if(r >= 0 && c >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];
			r = row;
			c = col - 1;
			shift = 4 * step + r * width + c;
			if(c >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			shift = row * width + col;
		    top_diff[base + shift] = temp;
                   bottom_diff[base + shift] += temp * filters[fbase + shift];	

		}
    }
	
    for(int col=0; col<width; col++){
		int location = base + (height-1)*width + col;
		int shift = fbase + (height-1)*width + col;
		bottom_diff[location] += top_diff[location] * filters[shift + step];
	}
	for (int row = 0; row < height; row++)
    {
      int location = base + row * width;
      int shift = fbase + row * width;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
	  
      location += width-1;
	  shift += width-1;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
      bottom_diff[location] += top_diff[location] * filters[shift + 4 * step];
    }

}

__global__ void nlf_right_backward(const int n, const float* filters, float* top_diff, const int channel, const int height, const int width, const int wsize,float* bottom_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (index >= n) {
        return;
    }
	
    int step = height * width;
    int base = index * step; //up->down
    int fbase = index/channel * step * wsize;
	for(int col = width - 1; col >= 0; col--){   
		for(int row = height-1; row >= 0; row --){
			float temp = top_diff[base + row * width + col];
//			int r = row;
//			int c = col;
//			int shift = 0 * step + row * width + col;
//			temp += top_data[base + r*width+c]*filters[fbase + shift];
			
			int r = row;
			int c = col + 1;
			int shift = 1 * step + r * width + c;
			if(c < width)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];
			
			r = row + 1;
			c = col + 1;
			shift = 2 * step + r * width + c;
			if(c < width && r < height)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];
			
			r = row - 1;
			c = col + 1;
			shift = 3 * step + r * width + c;
			if(c < width && r >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			r = row + 1;
                        c = col;
			shift = 4 * step + r * width + c;
			if(r < height)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			shift = row * width + col;
		    top_diff[base + shift] = temp;
            bottom_diff[base + shift] += temp * filters[fbase + shift];	
		}
    }
	
    for(int row=0; row<height; row++){
		int location = base + row*width;
		int shift = fbase + row*width;
		bottom_diff[location] += top_diff[location] * filters[shift + step];
	}
	for (int col = 0; col < width; col ++)
    {
      int location = base + col;
      int shift = fbase + col;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
      bottom_diff[location] += top_diff[location] * filters[shift + 4 * step];

	  
      location += (height - 1) * width;
	  shift += (height - 1) * width;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];

    }

}


__global__ void nlf_left_backward(const int n, const float* filters, float* top_diff, const int channel, const int height, const int width, const int wsize, float* bottom_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (index >= n) {
        return;
    }
	
    int step = height * width;
    int base = index * step; //up->down
    int fbase = index/channel * step * wsize;
	for(int col = 0; col < width; col ++){   
		for(int row = 0; row < height; row ++){
			float temp = top_diff[base + row * width + col];
//			int r = row;
//			int c = col;
//			int shift = 0 * step + row * width + col;
//			temp += top_data[base + r*width+c]*filters[fbase + shift];
			
			int r = row;
			int c = col - 1;
			int shift = 1 * step + r * width + c;
			if(c >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];
			
			r = row + 1;
			c = col - 1;
			shift = 2 * step + r * width + c;
			if(c >= 0 && r < height)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];
			
			r = row - 1;
			c = col - 1;
			shift = 3 * step + r * width + c;
			if(c >= 0 && r >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			r = row - 1;
			c = col;
			shift = 4 * step + r * width + c;
			if(r >= 0)
				temp += top_diff[base + r*width+c]*filters[fbase + shift];

			shift = row * width + col;
		    top_diff[base + shift] = temp;
            bottom_diff[base + shift] += temp * filters[fbase + shift];	
		}
    }
	

    for(int row=0; row<height; row++){
		int location = base + row*width + width-1;
		int shift = fbase + row*width + width-1;
		bottom_diff[location] += top_diff[location] * filters[shift + step];
	}
	for (int col = 0; col < width; col ++)
    {
      int location = base + col;
      int shift = fbase + col;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
	  
      location += (height - 1) * width;
	   shift += (height - 1) * width;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
      bottom_diff[location] += top_diff[location] * filters[shift + 4 * step];
    }

}

__global__ void nlf_filter_down_backward(const int n, const float* bottom_data, const float* top_data, const float* temp_diff, const int channel, const int height, const int width, const int wsize, float* filters_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step = height * width;
//    int base = index;
    int base = index/step*step*channel+index%step; //up->down
    int fbase = index/step*step*wsize+index%step;
    int row = index%step/width;
    int col = index%step%width;
    for(int i = 0; i < channel; i++){
        filters_diff[fbase] += temp_diff[base + i * step] * bottom_data[base + i * step];
	if(row - 1 >= 0)  
            filters_diff[fbase + step] += temp_diff[base + i*step] * top_data[base - width + i*step];
	else
		filters_diff[fbase + step] += temp_diff[base + i*step] * bottom_data[base + i*step];
	if(row - 1 >= 0 && col - 1 >= 0)
        filters_diff[fbase + 2*step] += temp_diff[base + i*step] * top_data[base - width - 1 + i*step];
        else
		filters_diff[fbase + 2*step] += temp_diff[base + i*step] * bottom_data[base + i*step];
	if(row - 1 >= 0 && col + 1 < width)
            filters_diff[fbase + 3*step] += temp_diff[base + i*step] * top_data[base - width + 1 + i*step];
        else
		filters_diff[fbase + 3*step] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(col - 1 >= 0)
            filters_diff[fbase + 4*step] += temp_diff[base + i*step] * top_data[base - 1 + i*step];
        else
		filters_diff[fbase + 4*step] += temp_diff[base + i*step] * bottom_data[base + i*step];
    }
		
}
__global__ void nlf_filter_up_backward(const int n, const float* bottom_data, const float* top_data, const float* temp_diff, const int channel, const int height, const int width, const int wsize, float* filters_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step = height * width;
//    int base = index;
    int base = index/step*step*channel+index%step; //up->down
    int fbase = index/step*step*wsize+index%step;
    int row = index%step/width;
    int col = index%step%width;
    for(int i=0; i< channel; i++){

        filters_diff[fbase] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(row + 1 < height)  
        filters_diff[fbase + step] += temp_diff[base + i*step] * top_data[base + width + i*step];
	else
		filters_diff[fbase + step] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(row + 1 < height && col - 1 >= 0)
        filters_diff[fbase + 2*step] += temp_diff[base + i*step] * top_data[base + width - 1 + i*step];
        else
		filters_diff[fbase + 2*step] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(row + 1 < height && col + 1 < width)
        filters_diff[fbase + 3*step] += temp_diff[base + i*step] * top_data[base + width + 1 + i*step];
        else
		filters_diff[fbase + 3*step] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(col + 1 < width)
            filters_diff[fbase + 4*step] += temp_diff[base + i*step] * top_data[base + 1 + i*step];
        else
		filters_diff[fbase + 4*step] += temp_diff[base + i*step] * bottom_data[base + i*step];
    }
}

__global__ void nlf_filter_right_backward(const int n, const float* bottom_data, const float* top_data, const float* temp_diff, const int channel, const int height, const int width, const int wsize, float* filters_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step = height * width;
//  int base = index;
    int base = index/step*step*channel+index%step; //up->down
    int fbase = index/step*step*wsize+index%step;
    int row = index%step/width;
    int col = index%step%width;
    for(int i=0; i < channel; i++){

        filters_diff[fbase] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(col - 1 >= 0)  
            filters_diff[fbase + step] += temp_diff[base + i*step] * top_data[base - 1   + i*step];
	else
	    filters_diff[fbase + step] += temp_diff[base + i*step] * bottom_data[base + i*step];
	
	if(col - 1 >= 0 && row - 1 >= 0)
            filters_diff[fbase + 2*step] += temp_diff[base + i*step] * top_data[base - width - 1 + i*step];
        else
	    filters_diff[fbase + 2*step] += temp_diff[base + i*step] * bottom_data[base + i*step];
	
	if(col -1 >= 0 && row + 1 < height)
            filters_diff[fbase + 3*step] += temp_diff[base + i*step] * top_data[base + width - 1 + i*step];
        else
	    filters_diff[fbase + 3*step] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(row - 1 >= 0)
            filters_diff[fbase + 4*step] += temp_diff[base + i*step] * top_data[base - width + i*step];
        else
	    filters_diff[fbase + 4*step] += temp_diff[base + i*step] * bottom_data[base + i*step];
    }
		
}

__global__ void nlf_filter_left_backward(const int n, const float* bottom_data, const float* top_data, const float* temp_diff, const int channel, const int height, const int width, const int wsize, float* filters_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int step = height * width;
//    int base = index;
    int base = index/step*step*channel+index%step; //up->down
    int fbase = index/step*step*wsize+index%step;
    int row = index%step/width;
    int col = index%step%width;
    for(int i = 0; i < channel; i++){
        filters_diff[fbase] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(col + 1 < width)  
        filters_diff[fbase + step] += temp_diff[base + i*step] * top_data[base + 1 + i*step];
	else
		filters_diff[fbase + step] += temp_diff[base + i*step] * bottom_data[base + i*step];
	
	if(col + 1 < width && row - 1 >= 0)
        filters_diff[fbase + 2*step] += temp_diff[base + i*step] * top_data[base - width + 1 + i*step];
        else
		filters_diff[fbase + 2*step] += temp_diff[base + i*step] * bottom_data[base + i*step];
	
	if(col + 1 < width && row + 1 < height)
        filters_diff[fbase + 3*step] += temp_diff[base + i*step] * top_data[base + width + 1 + i*step];
        else
		filters_diff[fbase + 3*step] += temp_diff[base + i*step] * bottom_data[base + i*step];

	if(row + 1 < height)
            filters_diff[fbase + 4*step] += temp_diff[base + i*step] * top_data[base + width + i*step];
        else
	    filters_diff[fbase + 4*step] += temp_diff[base + i*step] * bottom_data[base + i*step];

    }
		
}

void nlf_down_kernel_forward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down){
	int num = input.size(0);
	int channel = input.size(1);
	int height = input.size(2);
	int width = input.size(3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
       
	int wsize = guidance_down.size(1);
	float *top_down = output_down.data<float>();
	const float *bottom_data = input.data<float>();
	const float *g0 = guidance_down.data<float>();
	int n = num * channel;
	int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	int N = input.numel();
	cudaMemcpy (top_down, bottom_data, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_down_forward <<< threads, CUDA_NUM_THREADS >>> (n, g0, channel, height, width, wsize, top_down);
}
void nlf_up_kernel_forward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up){
	int num = input.size(0);
	int channel = input.size(1);
	int height = input.size(2);
	int width = input.size(3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
	int wsize = guidance_up.size(1);
	float *top_up = output_up.data<float>();
	const float *bottom_data = input.data<float>();
	const float *g1 = guidance_up.data<float>();
	int n = num * channel;
	int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	int N = input.numel();
	cudaMemcpy (top_up, bottom_data, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_up_forward <<< threads, CUDA_NUM_THREADS >>> (n, g1, channel, height, width, wsize, top_up);
}
void nlf_right_kernel_forward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right){
	int num = input.size(0);
	int channel = input.size(1);
	int height = input.size(2);
	int width = input.size(3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
	int wsize = guidance_right.size(1);
	float *top_right = output_right.data<float>();
	const float *bottom_data = input.data<float>();
	const float *g2 = guidance_right.data<float>();
	int n = num * channel;
	int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	int N = input.numel();
	cudaMemcpy (top_right, bottom_data, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_right_forward <<< threads, CUDA_NUM_THREADS >>> (n, g2, channel, height, width, wsize, top_right);
}
void nlf_left_kernel_forward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left){
	int num = input.size(0);
	int channel = input.size(1);
	int height = input.size(2);
	int width = input.size(3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
	int wsize = guidance_left.size(1);
	float *top_left = output_left.data<float>();
	const float *bottom_data = input.data<float>();
	const float *g3 = guidance_left.data<float>();
	int n = num * channel;
	int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	int N = input.numel();
	cudaMemcpy (top_left, bottom_data, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_left_forward <<< threads, CUDA_NUM_THREADS >>> (n, g3, channel, height, width, wsize, top_left);
}
void nlf_down_kernel_backward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down){
	int num = input.size (0);
	int channel = input.size (1);
	int height = input.size (2);
	int width = input.size (3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
	int wsize = guidance_down.size (1);
	const float *bottom_data = input.data<float>();
	float *grad_output = gradOutput.data<float>();
	float *grad_input = gradInput.data<float>();
	const float *top_down = output_down.data<float>();
	const float *g0 = guidance_down.data<float>();
	float *grad0 = grad_down.data<float>();
	
	int N = input.numel();
	int n = num * channel;
	cudaMemset (grad_input, 0, sizeof (float) * N);
	cudaMemset (grad0, 0, sizeof (float) * num*wsize*height*width );
	nlf_down_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g0, grad_output, channel, height, width, wsize, grad_input);
        n = num*height*width;
	nlf_filter_down_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, bottom_data, top_down, grad_output, channel, height, width, wsize, grad0);
}
void nlf_up_kernel_backward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_up){
	int num = input.size (0);
	int channel = input.size (1);
	int height = input.size (2);
	int width = input.size (3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
	int wsize = guidance_up.size (1);
	const float *bottom_data = input.data<float>();
	float *grad_output = gradOutput.data<float>();
	float *grad_input = gradInput.data<float>();
	const float *top_up = output_up.data<float>();
	const float *g1 = guidance_up.data<float>();
	float *grad1 = grad_up.data<float>();
	
	int N = input.numel();
	int n = num * channel;
	cudaMemset (grad_input, 0, sizeof (float) * N);
	cudaMemset (grad1, 0, sizeof (float) * num*wsize*height*width );
	nlf_up_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g1, grad_output, channel, height, width, wsize, grad_input);
        n = num*height*width;
	nlf_filter_up_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, bottom_data, top_up, grad_output, channel, height, width, wsize, grad1);
}
void nlf_right_kernel_backward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_right){
	int num = input.size (0);
	int channel = input.size (1);
	int height = input.size (2);
	int width = input.size (3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
	int wsize = guidance_right.size (1);
	const float *bottom_data = input.data<float>();
	float *grad_output = gradOutput.data<float>();
	float *grad_input = gradInput.data<float>();
	const float *top_right = output_right.data<float>();
	const float *g2 = guidance_right.data<float>();
	float *grad2 = grad_right.data<float>();
	
	int N = input.numel();
	int n = num * channel;
	cudaMemset (grad_input, 0, sizeof (float) * N);
	cudaMemset (grad2, 0, sizeof (float) * num*wsize*height*width );
	nlf_right_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g2, grad_output, channel, height, width, wsize, grad_input);
        n = num*height*width;
	nlf_filter_right_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, bottom_data, top_right, grad_output, channel, height, width, wsize, grad2);
}        
void nlf_left_kernel_backward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_left){
	int num = input.size (0);
	int channel = input.size (1);
	int height = input.size (2);
	int width = input.size (3);
        if(input.dim()>4){
	    channel = input.size(1) * input.size(2);
	    height = input.size(3);
	    width = input.size(4);
        }
	int wsize = guidance_left.size (1);
	const float *bottom_data = input.data<float>();
	float *grad_output = gradOutput.data<float>();
	float *grad_input = gradInput.data<float>();
	const float *top_left = output_left.data<float>();
	const float *g3 = guidance_left.data<float>();
	float *grad3 = grad_left.data<float>();
	
	int N = input.numel();
	int n = num * channel;
	cudaMemset (grad_input, 0, sizeof (float) * N);
	cudaMemset (grad3, 0, sizeof (float) * num*wsize*height*width );
	nlf_left_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g3, grad_output, channel, height, width, wsize, grad_input);
        n = num*height*width;
	nlf_filter_left_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, bottom_data, top_left, grad_output, channel, height, width, wsize, grad3);
}        

void nlf_kernel_forward (at::Tensor input, at::Tensor guidance_down,
                    at::Tensor guidance_up, at::Tensor guidance_right,
                    at::Tensor guidance_left, at::Tensor output_down,
		    at::Tensor output_up, at::Tensor output_right,
 		    at::Tensor output_left){

	int num = input.size(0);
	int channel = input.size(1);
	int height = input.size(2);
	int width = input.size(3);
	int wsize = guidance_down.size(1);

	float *top_down = output_down.data<float>();
	float *top_up = output_up.data<float>();
	float *top_right = output_right.data<float>();
	float *top_left = output_left.data<float>();

	const float *bottom_data = input.data<float>();
	
	const float *g0 = guidance_down.data<float>();
	const float *g1 = guidance_up.data<float>();
	const float *g2 = guidance_right.data<float>();
	const float *g3 = guidance_left.data<float>();

	int n = num * channel;
	int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	int N = input.numel();
//        printf("%d %d %d %d %d %d %d %d\n", num, channel, height, width, wsize, n, threads, N);
	cudaMemcpy (top_down, bottom_data, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_down_forward <<< threads, CUDA_NUM_THREADS >>> (n, g0, channel, height, width, wsize, top_down);
//        printf("sgf down done...\n");
	cudaMemcpy (top_up, top_down, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_up_forward <<< threads, CUDA_NUM_THREADS >>> (n, g1, channel, height, width, wsize, top_up);
//        printf("sgf up done...\n");
	
	cudaMemcpy (top_right, top_up, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_right_forward <<< threads, CUDA_NUM_THREADS >>> (n, g2, channel, height, width, wsize, top_right);

//        printf("sgf right done...\n");
	cudaMemcpy (top_left, top_right, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	nlf_left_forward <<< threads, CUDA_NUM_THREADS >>> (n, g3, channel, height, width, wsize, top_left);
//        printf("sgf left done...\n");
}


void nlf_kernel_backward (at::Tensor input, at::Tensor guidance_down,
                     at::Tensor guidance_up, at::Tensor guidance_right,
                     at::Tensor guidance_left, at::Tensor output_down,
         	     at::Tensor output_up, at::Tensor output_right,
		     at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down,
                     at::Tensor grad_up, at::Tensor grad_right,
                     at::Tensor grad_left){

	int num = input.size (0);
	int channel = input.size (1);
	int height = input.size (2);
	int width = input.size (3);
	int wsize = guidance_down.size (1);


	const float *bottom_data = input.data<float>();
	
	float *grad_output = gradOutput.data<float>();
	float *grad_input = gradInput.data<float>();
  
	const float *top_down = output_down.data<float>();
	const float *top_up = output_up.data<float>();
	const float *top_right = output_right.data<float>();
	const float *top_left = output_left.data<float>();
	const float *g0 = guidance_down.data<float>();
	const float *g1 = guidance_up.data<float>();
	const float *g2 = guidance_right.data<float>();
	const float *g3 = guidance_left.data<float>();

	float *grad0 = grad_down.data<float>();
	float *grad1 = grad_up.data<float>();
	float *grad2 = grad_right.data<float>();
	float *grad3 = grad_left.data<float>();

	
	
	int N = input.numel();
	int n = num * channel;
	
	cudaMemset (grad_input, 0, sizeof (float) * N);
	
	nlf_left_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g3, grad_output, channel, height, width, wsize, grad_input);
	nlf_filter_left_backward <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (N, bottom_data, top_left, grad_output, channel, height, width, wsize, grad3);
//        printf("backward left done...\n");
	
	cudaMemcpy (grad_output, grad_input, sizeof (float) * N, cudaMemcpyDeviceToDevice); 
	cudaMemset (grad_input, 0, sizeof (float) * N);
	nlf_right_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g2, grad_output, channel, height, width, wsize, grad_input);
	nlf_filter_right_backward <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (N, bottom_data, top_right, grad_output, channel, height, width, wsize, grad2);
//        printf("backward right done...\n");
	
	
	cudaMemcpy (grad_output, grad_input, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	cudaMemset (grad_input, 0, sizeof (float) * N);
	nlf_up_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g1, grad_output, channel, height, width, wsize, grad_input);
	nlf_filter_up_backward <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (N, bottom_data, top_up, grad_output, channel, height, width, wsize, grad1);
	
	cudaMemcpy (grad_output, grad_input, sizeof (float) * N, cudaMemcpyDeviceToDevice);
	cudaMemset (grad_input, 0, sizeof (float) * N);
	nlf_down_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (n, g0, grad_output, channel, height, width, wsize, grad_input);
	nlf_filter_down_backward <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, CUDA_NUM_THREADS >>> (N, bottom_data, top_down, grad_output, channel, height, width, wsize, grad0);
}	



 
#ifdef __cplusplus
    }
#endif
