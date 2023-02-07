//#include <torch/torch.h>
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



__global__ void Max (const int n, const float *top_temp, float *top_data, float *mask,
     const int mask_index){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  if (top_data[index] < top_temp[index])
    {
      top_data[index] = top_temp[index];
      mask[index] = mask_index;
    }
}

__global__ void get_temp_grad (const int n, const float *gradOutput, const float *mask,
	       float *top_grad, const int mask_index){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  if (((int) mask[index]) == mask_index)
    top_grad[index] = gradOutput[index];
}

__global__ void MaxDepth (const int n, const float *bottom_data, const int step,
	  const int depth, float *idx){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int base = index / step * step * depth + index % step;
  int k = 0;
  for (int i = 1; i < depth; i++)
    if (bottom_data[base + k * step] < bottom_data[base + i * step])
      k = i;
  idx[index] = k;
}

__global__ void sga_down_forward (const int n, const float *filters, const int height,
		  const int width, const int depth, const int wsize,
		  float *top_data){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
//  int wsize=radius+1;
  int base = index / width * step * depth + index % width;	//up->down
  int fbase = index / width * step * wsize + index % width;

  int kp = 0;

  for (int row = 0; row < height; row++)
    {
      int shift = fbase + row * width;

      int base0 = base + row * width;
      int k = kp;
      kp = 0;

/*        if(row-1>=0)
            for(int i = 1; i < depth; i++){
	        if(top_data[base0-width+k*step]<top_data[base0-width+i*step])
		    k = i;
*/
      for (int d = 0; d < depth; d++)
	{
	  float temp = 0;
	  int location = base0 + d * step;
	  temp += top_data[location] * filters[shift];
	  if (row - 1 >= 0)
	    temp += top_data[location - width] * filters[shift + step];
	  else
	    temp += top_data[location] * filters[shift + step];

	  if (row - 1 >= 0 && d - 1 >= 0)
	    temp +=
	      top_data[location - width - step] * filters[shift + 2 * step];
	  else
	    temp += top_data[location] * filters[shift + 2 * step];
	  if (row - 1 >= 0 && d + 1 < depth)
	    temp +=
	      top_data[location - width + step] * filters[shift + 3 * step];
	  else
	    temp += top_data[location] * filters[shift + 3 * step];
	  if (row - 1 >= 0)
	    temp +=
	      top_data[base0 - width + k * step] * filters[shift + 4 * step];
	  else
	    temp += top_data[location] * filters[shift + 4 * step];
	  top_data[location] = temp;

	  if (top_data[base0 + kp * step] < temp)
	    kp = d;

	}
    }
}

__global__ void sga_down_data_backward (const int n, const float *filters, float *top_diff,
			const float *idx, const int height, const int width,
			const int depth, const int wsize, float *bottom_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / width * step * depth + index % width;	//up->down
  int fbase = index / width * step * wsize + index % width;
//1
  int base_idx = index / width * step + index % width;
//
  for (int row = height - 1; row >= 0; row--)
    {
      int shift = fbase + row * width;
      for (int d = 0; d < depth; d++)
	{
	  int location = base + d * step + row * width;
	  float temp = top_diff[location];
	  if (row + 1 < height)
	    temp +=
	      top_diff[location + width] * filters[shift + width + step];

	  if (row + 1 < height && d + 1 < depth)
	    temp +=
	      top_diff[location + width + step] * filters[shift + width +
							  2 * step];
	  if (row + 1 < height && d - 1 >= 0)
	    temp +=
	      top_diff[location + width - step] * filters[shift + width +
							  3 * step];
	  top_diff[location] = temp;
	  bottom_diff[location] += temp * filters[shift];
	}
//2
      if (row + 1 < height)
	{
	  int k = idx[base_idx + row * width];
	  int location = base + k * step + row * width;
	  float temp = 0;
	  for (int d = 0; d < depth; d++)
	    temp +=
	      top_diff[base + row * width + width +
		       d * step] * filters[shift + width + 4 * step];
	  top_diff[location] += temp;
	  bottom_diff[location] += temp * filters[shift];
	}
//2

    }

/*	for(int d = 0; d < depth; d ++){
		int shift = fbase;
		int location = base + d * step;
		bottom_diff[location] += top_diff[location] * (filters[shift + step] + filters[shift + 2*step] + filters[shift + 3*step] + filters[shift + 4*step]); 
 //       bottom_diff[location] += top_diff[location];
		shift += width;
		location += width;
		bottom_diff[location] += top_diff[location] * filters[shift + 2*step];	
	}
	for(int row=1;row<height;row++){
		int location = base + row * width;
		int shift = fbase + row * width;
		bottom_diff[location] += top_diff[location] * filters[shift + 3*step]; 
		location += (depth - 1)*step;
		bottom_diff[location] += top_diff[location] * filters[shift + 4*step]; 
	}
*/
  for (int row = 0; row < height; row++)
    {
      int location = base + row * width;
      int shift = fbase + row * width;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
      location += (depth - 1) * step;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
    }
}

__global__ void sga_down_weight_backward (const int n, const float *bottom_data,
			  const float *top_data, const float *temp_diff,
			  const float *idx, const int height, const int width,
			  const int depth, const int wsize,
			  float *filters_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / step * step * depth + index % step;	//up->down
  int fbase = index / step * step * wsize + index % step;

  int row = index % step / width;
  for (int i = 0; i < depth; i++)
    filters_diff[fbase] +=
      temp_diff[base + i * step] * bottom_data[base + i * step];
  if (row - 1 >= 0)
    {
      int location = fbase + step;
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + i * step - width];

      location = fbase + 2 * step;
      filters_diff[location] += temp_diff[base] * bottom_data[base];
      for (int i = 1; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i - 1) * step -
						width];

      location = fbase + 3 * step;
      filters_diff[location] +=
	temp_diff[base + (depth - 1) * step] * bottom_data[base +
							   (depth -
							    1) * step];
      for (int i = 0; i < depth - 1; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i + 1) * step -
						width];
    }
/*
    else{
		for(int i=0; i<depth; i++){
			float temp = temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		    filters_diff[fbase + 3*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + 4*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		}

	}
*/
//1
  if (row - 1 >= 0)
    {
      int location = fbase + 4 * step;
      int k = idx[index - width];
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + k * step - width];
    }
//
/*
    else{
		int location = fbase + 2*step;
		for(int i=0; i<depth; i++)
			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
	}
*/
}



__global__ void sga_up_forward (const int n, const float *filters, const int height,
		const int width, const int depth, const int wsize,
		float *top_data){

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= n)
    {
      return;
    }
  int step = height * width;
  //   int wsize=radius+1;

  int base = index / width * step * depth + index % width;	//up->down
  int fbase = index / width * step * wsize + index % width;

  int kp = 0;			//1

  for (int row = height - 1; row >= 0; row--)
    {
      int shift = fbase + row * width;
//2
      int base0 = base + row * width;
      int k = kp;
      kp = 0;
//2
      for (int d = 0; d < depth; d++)
	{
	  float temp = 0;
	  int location = base + d * step + row * width;
	  temp += top_data[location] * filters[shift];
	  if (row + 1 < height)
	    temp += top_data[location + width] * filters[shift + step];
	  else
	    temp += top_data[location] * filters[shift + step];

	  if (row + 1 < height && d - 1 >= 0)
	    temp +=
	      top_data[location + width - step] * filters[shift + 2 * step];
	  else
	    temp += top_data[location] * filters[shift + 2 * step];
	  if (row + 1 < height && d + 1 < depth)
	    temp +=
	      top_data[location + width + step] * filters[shift + 3 * step];
	  else
	    temp += top_data[location] * filters[shift + 3 * step];

//3
	  if (row + 1 < height)
	    temp +=
	      top_data[base0 + width + k * step] * filters[shift + 4 * step];
	  else
	    temp += top_data[location] * filters[shift + 4 * step];
	  top_data[location] = temp;

	  if (top_data[base0 + kp * step] < temp)
	    kp = d;
//3

	}
    }
}

__global__ void sga_up_data_backward (const int n, const float *filters, float *top_diff,
		      const float *idx, const int height, const int width,
		      const int depth, const int wsize, float *bottom_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / width * step * depth + index % width;	//up->down
  int fbase = index / width * step * wsize + index % width;

//1
  int base_idx = index / width * step + index % width;
//
  for (int row = 0; row < height; row++)
    {
      int shift = fbase + row * width;
      for (int d = 0; d < depth; d++)
	{
	  int location = base + d * step + row * width;
	  float temp = top_diff[location];
	  if (row - 1 >= 0)
	    temp +=
	      top_diff[location - width] * filters[shift - width + step];
	  if (row - 1 >= 0 && d + 1 < depth)
	    temp +=
	      top_diff[location - width + step] * filters[shift - width +
							  2 * step];
	  if (row - 1 >= 0 && d - 1 >= 0)
	    temp +=
	      top_diff[location - width - step] * filters[shift - width +
							  3 * step];
	  top_diff[location] = temp;
	  bottom_diff[location] += temp * filters[shift];
	}

//2
      if (row - 1 >= 0)
	{
	  int k = idx[base_idx + row * width];
	  int location = base + k * step + row * width;
	  float temp = 0;
	  for (int d = 0; d < depth; d++)
	    temp +=
	      top_diff[base + row * width - width +
		       d * step] * filters[shift - width + 4 * step];
	  top_diff[location] += temp;
	  bottom_diff[location] += temp * filters[shift];
	}
//2
    }

/*	for(int d = 0; d < depth; d ++){
		int shift = fbase + width*(height-1);
		int location = base + width*(height-1) + d * step;
		bottom_diff[location] += top_diff[location] * (filters[shift + step] + filters[shift + 2*step] + filters[shift + 3*step] + filters[shift + 4*step]); 
//        bottom_diff[location] += top_diff[location];
		shift -= width;
		location -= width;
		bottom_diff[location] += top_diff[location] * filters[shift + 2*step];	
	}
	for(int row=0;row<height-1;row++){
		int shift = fbase + row * width;
		int location = base + row * width;
		bottom_diff[location] += top_diff[location] * filters[shift + 3*step]; 
		location += (depth - 1)*step;
		bottom_diff[location] += top_diff[location] * filters[shift + 4*step]; 
	}*/
  for (int row = 0; row < height; row++)
    {
      int shift = fbase + row * width;
      int location = base + row * width;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
      location += (depth - 1) * step;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
    }
}

__global__ void sga_up_weight_backward (const int n, const float *bottom_data,
			const float *top_data, const float *temp_diff,
			const float *idx, const int height, const int width,
			const int depth, const int wsize, float *filters_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / step * step * depth + index % step;	//up->down
  int fbase = index / step * step * wsize + index % step;

  int row = index % step / width;
  for (int i = 0; i < depth; i++)
    filters_diff[fbase] +=
      temp_diff[base + i * step] * bottom_data[base + i * step];
  if (row + 1 < height)
    {
      int location = fbase + step;
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + i * step + width];

      location = fbase + 2 * step;
      filters_diff[location] += temp_diff[base] * bottom_data[base];
      for (int i = 1; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i - 1) * step +
						width];

      location = fbase + 3 * step;
      filters_diff[location] +=
	temp_diff[base + (depth - 1) * step] * bottom_data[base +
							   (depth -
							    1) * step];
      for (int i = 0; i < depth - 1; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i + 1) * step +
						width];
    }
/*
    else{
		//int location = fbase + step;
		for(int i=0; i<depth; i++){
			float temp = temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		    filters_diff[fbase + 3*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + 4*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		}
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];		
//		location = fbase + 3*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
//		
//		location = fbase + 4*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
	}*/
//1
  if (row + 1 < height)
    {
      int location = fbase + 4 * step;
      int k = idx[index + width];
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + k * step + width];
    }
//

/*
    else{
		int location = fbase + 2*step;
		for(int i=0; i<depth; i++)
			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
	}*/
}

__global__ void sga_right_forward (const int n, const float *filters, const int height,
		   const int width, const int depth, const int wsize,
		   float *top_data){

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= n)
    {
      return;
    }
  int step = height * width;
  //   int wsize=radius+1;

  int base = index / height * step * depth + (index % height) * width;	//up->down
  int fbase = index / height * step * wsize + (index % height) * width;

  int kp = 0;

  for (int col = 0; col < width; col++)
    {
      int shift = fbase + col;
//2
      int base0 = base + col;
      int k = kp;
      kp = 0;
//2
      for (int d = 0; d < depth; d++)
	{
	  float temp = 0;
	  int location = base + d * step + col;
	  temp += top_data[location] * filters[shift];
	  if (col - 1 >= 0)
	    temp += top_data[location - 1] * filters[shift + step];
	  else
	    temp += top_data[location] * filters[shift + step];

	  if (col - 1 >= 0 && d - 1 >= 0)
	    temp += top_data[location - 1 - step] * filters[shift + 2 * step];
	  else
	    temp += top_data[location] * filters[shift + 2 * step];
	  if (col - 1 >= 0 && d + 1 < depth)
	    temp += top_data[location - 1 + step] * filters[shift + 3 * step];
	  else
	    temp += top_data[location] * filters[shift + 3 * step];

//3
	  if (col - 1 >= 0)
	    temp +=
	      top_data[base0 - 1 + k * step] * filters[shift + 4 * step];
	  else
	    temp += top_data[location] * filters[shift + 4 * step];
	  top_data[location] = temp;

	  if (top_data[base0 + kp * step] < temp)
	    kp = d;
//3
	}
    }
}

__global__ void sga_right_data_backward (const int n, const float *filters, float *top_diff,
			 const float *idx, const int height, const int width,
			 const int depth, const int wsize, float *bottom_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / height * step * depth + (index % height) * width;	//up->down
  int fbase = index / height * step * wsize + (index % height) * width;
//1
  int base_idx = index / height * step + (index % height) * width;
//
  for (int col = width - 1; col >= 0; col--)
    {
      int shift = fbase + col;
      for (int d = 0; d < depth; d++)
	{
	  int location = base + d * step + col;
	  float temp = top_diff[location];
	  if (col + 1 < width)
	    temp += top_diff[location + 1] * filters[shift + 1 + step];
	  if (col + 1 < width && d + 1 < depth)
	    temp +=
	      top_diff[location + 1 + step] * filters[shift + 1 + 2 * step];
	  if (col + 1 < width && d - 1 >= 0)
	    temp +=
	      top_diff[location + 1 - step] * filters[shift + 1 + 3 * step];
	  top_diff[location] = temp;
	  bottom_diff[location] += (temp * filters[shift]);
	}
//2
      if (col + 1 < width)
	{
	  int k = idx[base_idx + col];
	  int location = base + k * step + col;
	  float temp = 0;
	  for (int d = 0; d < depth; d++)
	    temp +=
	      top_diff[base + col + 1 + d * step] * filters[shift + 1 +
							    4 * step];
	  top_diff[location] += temp;
	  bottom_diff[location] += temp * filters[shift];
	}
//2     
    }
/*
	for(int d = 0; d < depth; d ++){
		int shift = fbase;// + width*(height-1);
		int location = base;// + width*(height-1) + d * step;
		bottom_diff[location] += top_diff[location] * (filters[shift + step] + filters[shift + 2*step] + filters[shift + 3*step] + filters[shift + 4*step]);
 //       bottom_diff[location] += top_diff[location];
		shift += 1;
		location += 1;
		bottom_diff[location] += top_diff[location] * filters[shift + 2*step];	
	}
	for(int col=1;col<width;col++){
		int shift = fbase + col;
		int location = base + col;
		bottom_diff[location] += top_diff[location] * filters[shift + 3*step]; 
		location += (depth - 1)*step;
		bottom_diff[location] += top_diff[location] * filters[shift + 4*step]; 
	}*/
  for (int col = 0; col < width; col++)
    {
      int shift = fbase + col;
      int location = base + col;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
      location += (depth - 1) * step;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
    }
}

__global__ void sga_right_weight_backward (const int n, const float *bottom_data,
			   const float *top_data, const float *temp_diff,
			   const float *idx, const int height,
			   const int width, const int depth, const int wsize,
			   float *filters_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / step * step * depth + index % step;	//up->down
  int fbase = index / step * step * wsize + index % step;

  //   int row = index%step/width;
  int col = index % step % width;
  for (int i = 0; i < depth; i++)
    filters_diff[fbase] +=
      temp_diff[base + i * step] * bottom_data[base + i * step];
  if (col - 1 >= 0)
    {
      int location = fbase + step;
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + i * step - 1];

      location = fbase + 2 * step;
      filters_diff[location] += temp_diff[base] * bottom_data[base];
      for (int i = 1; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i - 1) * step - 1];

      location = fbase + 3 * step;
      filters_diff[location] +=
	temp_diff[base + (depth - 1) * step] * bottom_data[base +
							   (depth -
							    1) * step];
      for (int i = 0; i < depth - 1; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i + 1) * step - 1];
    }
/*
    else{
		//int location = fbase + step;
		for(int i=0; i<depth; i++){
			float temp = temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		    filters_diff[fbase + 3*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + 4*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		}
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];		
//		location = fbase + 3*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
//		
//		location = fbase + 4*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
	}*/
//1
  if (col - 1 >= 0)
    {
      int location = fbase + 4 * step;
      int k = idx[index - 1];
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + k * step - 1];
    }
//
/*
    else{
		int location = fbase + 2*step;
		for(int i=0; i<depth; i++)
			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
	}*/
}

__global__ void sga_left_forward (const int n, const float *filters, const int height,
		  const int width, const int depth, const int wsize,
		  float *top_data){

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= n)
    {
      return;
    }
  int step = height * width;
  //   int wsize=radius+1;

  int base = index / height * step * depth + (index % height) * width;	//up->down
  int fbase = index / height * step * wsize + (index % height) * width;

  int kp = 0;

  for (int col = width - 1; col >= 0; col--)
    {
      int shift = fbase + col;
//2
      int base0 = base + col;
      int k = kp;
      kp = 0;
//2
      for (int d = 0; d < depth; d++)
	{
	  float temp = 0;
	  int location = base + d * step + col;
	  temp += top_data[location] * filters[shift];
	  if (col + 1 < width)
	    temp += top_data[location + 1] * filters[shift + step];
	  else
	    temp += top_data[location] * filters[shift + step];

	  if (col + 1 < width && d - 1 >= 0)
	    temp += top_data[location + 1 - step] * filters[shift + 2 * step];
	  else
	    temp += top_data[location] * filters[shift + 2 * step];
	  if (col + 1 < width && d + 1 < depth)
	    temp += top_data[location + 1 + step] * filters[shift + 3 * step];
	  else
	    temp += top_data[location] * filters[shift + 3 * step];

//3
	  if (col + 1 < width)
	    temp +=
	      top_data[base0 + 1 + k * step] * filters[shift + 4 * step];
	  else
	    temp += top_data[location] * filters[shift + 4 * step];
	  top_data[location] = temp;

	  if (top_data[base0 + kp * step] < temp)
	    kp = d;
//3
	}
    }
}

__global__ void sga_left_data_backward (const int n, const float *filters, float *top_diff,
			const float *idx, const int height, const int width,
			const int depth, const int wsize, float *bottom_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / height * step * depth + (index % height) * width;	//up->down
  int fbase = index / height * step * wsize + (index % height) * width;
//1
  int base_idx = index / height * step + (index % height) * width;
//
  for (int col = 0; col < width; col++)
    {
      int shift = fbase + col;
      for (int d = 0; d < depth; d++)
	{
	  int location = base + d * step + col;
	  float temp = top_diff[location];
	  if (col - 1 >= 0)
	    temp += top_diff[location - 1] * filters[shift - 1 + step];
	  if (col - 1 >= 0 && d + 1 < depth)
	    temp +=
	      top_diff[location - 1 + step] * filters[shift - 1 + 2 * step];
	  if (col - 1 >= 0 && d - 1 >= 0)
	    temp +=
	      top_diff[location - 1 - step] * filters[shift - 1 + 3 * step];
	  top_diff[location] = temp;
	  bottom_diff[location] += temp * filters[shift];
	}
//2
      if (col - 1 >= 0)
	{
	  int k = idx[base_idx + col];
	  int location = base + k * step + col;
	  float temp = 0;
	  for (int d = 0; d < depth; d++)
	    temp +=
	      top_diff[base + col - 1 + d * step] * filters[shift - 1 +
							    4 * step];
	  top_diff[location] += temp;
//top_diff[base + col - 1 + d*step] * filters[shift - 1 + 4*step];
	  bottom_diff[location] += temp * filters[shift];
	}
//2             
    }
/*
	for(int d = 0; d < depth; d ++){
		int shift = fbase + width-1;// + width*(height-1);
		int location = base + width-1;// + width*(height-1) + d * step;
		bottom_diff[location] += top_diff[location] * (filters[shift + step] + filters[shift + 2*step] + filters[shift + 3*step] + filters[shift + 4*step]); 
//        bottom_diff[location] += top_diff[location];
		shift -= 1;
		location -= 1;
		bottom_diff[location] += top_diff[location] * filters[shift + 2*step];	
	}
	for(int col=0;col<width-1;col++){
		int shift = fbase + col;
		int location = base + col;
		bottom_diff[location] += top_diff[location] * filters[shift + 3*step]; 
		location += (depth - 1)*step;
		bottom_diff[location] += top_diff[location] * filters[shift + 4*step]; 
	}*/
  for (int col = 0; col < width; col++)
    {
      int shift = fbase + col;
      int location = base + col;
      bottom_diff[location] += top_diff[location] * filters[shift + 2 * step];
      location += (depth - 1) * step;
      bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
    }
}

__global__ void sga_left_weight_backward (const int n, const float *bottom_data,
			  const float *top_data, const float *temp_diff,
			  const float *idx, const int height, const int width,
			  const int depth, const int wsize,
			  float *filters_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int base = index / step * step * depth + index % step;	//up->down
  int fbase = index / step * step * wsize + index % step;

  //   int row = index%step/width;
  int col = index % step % width;
  for (int i = 0; i < depth; i++)
    filters_diff[fbase] +=
      temp_diff[base + i * step] * bottom_data[base + i * step];
  if (col + 1 < width)
    {
      int location = fbase + step;
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + i * step + 1];

      location = fbase + 2 * step;
      filters_diff[location] += temp_diff[base] * bottom_data[base];
      for (int i = 1; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i - 1) * step + 1];

      location = fbase + 3 * step;
      filters_diff[location] +=
	temp_diff[base + (depth - 1) * step] * bottom_data[base +
							   (depth -
							    1) * step];
      for (int i = 0; i < depth - 1; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + (i + 1) * step + 1];
    }
/*
    else{
		//int location = fbase + step;
		for(int i=0; i<depth; i++){
			float temp = temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		    filters_diff[fbase + 3*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
			filters_diff[fbase + 4*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
		}
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];		
//		location = fbase + 3*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
//		
//		location = fbase + 4*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
	}*/
//1
  if (col + 1 < width)
    {
      int location = fbase + 4 * step;
      int k = idx[index + 1];
      for (int i = 0; i < depth; i++)
	filters_diff[location] +=
	  temp_diff[base + i * step] * top_data[base + k * step + 1];
    }
//
/*
    else{
		int location = fbase + 2*step;
		for(int i=0; i<depth; i++)
			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
	}
*/
}

void sga_kernel_forward (at::Tensor input, at::Tensor guidance_down,
		    at::Tensor guidance_up, at::Tensor guidance_right,
		    at::Tensor guidance_left, at::Tensor temp_out,
		    at::Tensor output, at::Tensor mask){

  int num = input.size(0);
  int channel = input.size(1);
  int depth = input.size(2);
  int height = input.size(3);
  int width = input.size(4);
  int wsize = guidance_down.size(2);

  //THCudaTensor_nElement(state, input);
  float *top_data = output.data<float>();
  float *top_temp = temp_out.data<float>();
  float *top_mask = mask.data<float>();

  const float *bottom_data = input.data<float>();
  const float *g0 = guidance_down.data<float>();
  const float *g1 = guidance_up.data<float>();
  const float *g2 = guidance_right.data<float>();
  const float *g3 = guidance_left.data<float>();

  int n = num * channel * width;
  int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  int N = input.numel ();
//      cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
  cudaMemcpy (top_temp, bottom_data, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);
  sga_down_forward <<< threads, CUDA_NUM_THREADS >>> (n, g0, height, width,
						      depth, wsize, top_temp);
//      cudaMemset( top_mask, 0, sizeof(float)*N);
  cudaMemcpy (top_data, top_temp, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);

  cudaMemcpy (top_temp, bottom_data, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);
  sga_up_forward <<< threads, CUDA_NUM_THREADS >>> (n, g1, height, width,
						    depth, wsize, top_temp);

  Max <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, top_temp, top_data, top_mask, 1);

  n = num * channel * height;
  threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

  cudaMemcpy (top_temp, bottom_data, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);
  sga_right_forward <<< threads, CUDA_NUM_THREADS >>> (n, g2, height, width,
						       depth, wsize,
						       top_temp);
  Max <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, top_temp, top_data, top_mask, 2);

  cudaMemcpy (top_temp, bottom_data, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);
  sga_left_forward <<< threads, CUDA_NUM_THREADS >>> (n, g3, height, width,
						      depth, wsize, top_temp);
  Max <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, top_temp, top_data, top_mask, 3);

//      cudaMemset( top_temp, 0, sizeof(float)*THCudaTensor_nElement(state, top_temp));

}

void sga_kernel_backward (at::Tensor input, at::Tensor guidance_down,
		     at::Tensor guidance_up, at::Tensor guidance_right,
		     at::Tensor guidance_left, at::Tensor temp_out,
		     at::Tensor mask, at::Tensor max_idx,
		     at::Tensor gradOutput, at::Tensor temp_grad,
		     at::Tensor gradInput, at::Tensor grad_down,
		     at::Tensor grad_up, at::Tensor grad_right,
		     at::Tensor grad_left){

  int num = input.size(0);
  int channel = input.size(1);
  int depth = input.size(2);
  int height = input.size(3);
  int width = input.size(4);
  int wsize = guidance_down.size(2);

  //THCudaTensor_nElement(state, input);
  float *top_grad = temp_grad.data<float>();
  float *top_temp = temp_out.data<float>();
  const float *top_mask = mask.data<float>();

  const float *bottom_data = input.data<float>();
  const float *grad_out = gradOutput.data<float>();

  const float *g0 = guidance_down.data<float>();
  const float *g1 = guidance_up.data<float>();
  const float *g2 = guidance_right.data<float>();
  const float *g3 = guidance_left.data<float>();

  float *grad0 = grad_down.data<float>();
  float *grad1 = grad_up.data<float>();
  float *grad2 = grad_right.data<float>();
  float *grad3 = grad_left.data<float>();
  float *grad_input = gradInput.data<float>();

  float *idx = max_idx.data<float>();

  int N = input.numel ();
//      cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

//backward for left             
  int n = num * channel * height;
//              cudaMemcpy(top_temp, bottom_data, sizeof(float)*N, cudaMemcpyDeviceToDevice);
//              sga_left_forward<<<(n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>
//              (n,g3,height,width,depth,wsize,top_temp);

  cudaMemset (top_grad, 0, sizeof (float) * N);
  get_temp_grad <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, grad_out, top_mask, top_grad, 3);

  N = num * channel * width * height;
  MaxDepth <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, top_temp, height * width, depth, idx);

  sga_left_data_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, g3, top_grad, idx, height, width, depth, wsize,
			  grad_input);
  n = num * channel * width * height;
  sga_left_weight_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, bottom_data, top_temp, top_grad, idx, height,
			  width, depth, wsize, grad3);
//backward for down             
  N = input.numel ();
  n = num * channel * width;
  cudaMemcpy (top_temp, bottom_data, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);
  sga_down_forward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, g0, height, width, depth, wsize, top_temp);

  cudaMemset (top_grad, 0, sizeof (float) * N);
  get_temp_grad <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, grad_out, top_mask, top_grad, 0);

  N = num * channel * width * height;
  MaxDepth <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, top_temp, height * width, depth, idx);

  sga_down_data_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, g0, top_grad, idx, height, width, depth, wsize,
			  grad_input);
  n = num * channel * width * height;
  sga_down_weight_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, bottom_data, top_temp, top_grad, idx, height,
			  width, depth, wsize, grad0);
// backward for up              
  N = input.numel ();
  n = num * channel * width;
  cudaMemcpy (top_temp, bottom_data, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);
  sga_up_forward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, g1, height, width, depth, wsize, top_temp);

  cudaMemset (top_grad, 0, sizeof (float) * N);
  get_temp_grad <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, grad_out, top_mask, top_grad, 1);
  N = num * channel * width * height;
  MaxDepth <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, top_temp, height * width, depth, idx);

  sga_up_data_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, g1, top_grad, idx, height, width, depth, wsize,
			  grad_input);
  n = num * channel * width * height;
  sga_up_weight_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, bottom_data, top_temp, top_grad, idx, height,
			  width, depth, wsize, grad1);
//backward for right            
  N = input.numel ();
  n = num * channel * height;
  cudaMemcpy (top_temp, bottom_data, sizeof (float) * N,
	      cudaMemcpyDeviceToDevice);
  sga_right_forward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, g2, height, width, depth, wsize, top_temp);

  cudaMemset (top_grad, 0, sizeof (float) * N);
  get_temp_grad <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, grad_out, top_mask, top_grad, 2);

  N = num * channel * width * height;
  MaxDepth <<< (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (N, top_temp, height * width, depth, idx);

  sga_right_data_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, g2, top_grad, idx, height, width, depth, wsize,
			  grad_input);
  n = num * channel * width * height;
  sga_right_weight_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, bottom_data, top_temp, top_grad, idx, height,
			  width, depth, wsize, grad2);
}

__global__ void lga_filtering_forward (const int n, const float *bottom_data,
		       const float *filters, const int height,
		       const int width, const int channel, const int radius,
		       float *top_data){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("OK\n");
//    printf("%d, %.2f, %.2f\n", index, bottom_data[index], top_data[index]);
  if (index >= n)
    {
      return;
    }
//    top_data[index]=1.0;
//    assert(0);
  int step = height * width;
  int wsize = 2 * radius + 1;
//      int fsize=wsize*wsize*3;
  int fbase =
    index / (step * channel) * (step * wsize * wsize * 3) + index % step;
  int row = index % step / width;
  int col = index % width;
  int depth = index / step % channel;
  for (int d = -1; d <= 1; d++)
    {
      for (int r = -radius; r <= radius; r++)
	{
	  for (int c = -radius; c <= radius; c++)
	    {
	      int rr = r + row;
	      int cc = c + col;
	      int dd = d + depth;
	      int shift = 0;
	      if (rr >= 0 && cc >= 0 && dd >= 0 && rr < height && cc < width
		  && dd < channel)
		shift = r * width + c + d * step;
	      int location =
		(d + 1) * (wsize * wsize) + (r + radius) * wsize + c + radius;
	      top_data[index] +=
		bottom_data[index + shift] * filters[fbase + location * step];
	    }
	}
    }
//        top_data[index]=1.0;
//        printf("%d, %d, %d, %.2f, %.2f\n", index, row, col, bottom_data[index], top_data[index]);
}

__global__ void lga_filter_backward (const int n, const float *bottom_data,
		     const float *top_diff, const int height, const int width,
		     const int channel, const int radius, float *filter_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int wsize = 2 * radius + 1;

  int base =
    index / (step * wsize * wsize * 3) * (step * channel) + index % step;
  int location = index / step % (wsize * wsize * 3);
  int d = location / (wsize * wsize) - 1;
  int r = (location / wsize) % wsize - radius;
  int c = location % wsize - radius;

  int rr = index % step / width + r;
  int cc = index % width + c;

  for (int i = 0; i < channel; i++)
    {
      int dd = i + d;
      if (rr >= 0 && cc >= 0 && dd >= 0 && rr < height && cc < width
	  && dd < channel)
	{
	  int shift = r * width + c + d * step;
	  filter_diff[index] +=
	    top_diff[base + i * step] * bottom_data[base + shift + i * step];
	}
      else
	filter_diff[index] +=
	  top_diff[base + i * step] * bottom_data[base + i * step];
    }


}

__global__ void lga_data_backward (const int n, const float *filters, const float *top_diff,
		   const int height, const int width, const int channel,
		   const int radius, float *bottom_diff){

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= n)
    {
      return;
    }
  int step = height * width;
  int wsize = 2 * radius + 1;
//      int fsize=wsize*wsize*3;
  int fbase =
    index / (step * channel) * (step * wsize * wsize * 3) + index % step;
  int row = index % step / width;
  int col = index % width;
  int depth = index / step % channel;
  for (int d = -1; d <= 1; d++)
    {
      for (int r = -radius; r <= radius; r++)
	{
	  for (int c = -radius; c <= radius; c++)
	    {
	      int rr = r + row;
	      int cc = c + col;
	      int dd = d + depth;
	      //      int shift = 0;
	      if (rr >= 0 && cc >= 0 && dd >= 0 && rr < height && cc < width
		  && dd < channel)
		{
		  int shift = r * width + c + d * step;
		  //      int fshift= r*width+c;
		  int location =
		    (-d + 1) * (wsize * wsize) + (-r + radius) * wsize - c +
		    radius;
		  bottom_diff[index] +=
		    top_diff[index + shift] * filters[fbase + r * width + c +
						      location * step];
		}
	      else
		{
		  int location =
		    (d + 1) * (wsize * wsize) + (r + radius) * wsize + c +
		    radius;
		  bottom_diff[index] +=
		    top_diff[index] * filters[fbase + location * step];
		}
	    }
	}
    }
}

void lga_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
	     const int radius){

//        print_kernel<<<10, 10>>>();
//        cudaDeviceSynchronize();
  //       int num=input->size(0);
  int channel = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
  int n = input.numel ();
  //       printf("%d, %d, %d, %d, %d\n", height, width, channel, n, radius);
  //       cudaStream_t stream = at::cuda::getCurrentCUDAStream();
/*        float *temp = new float[n];
        float *out = input.data<float>();
        cudaMemcpy(temp,out,n*sizeof(float),cudaMemcpyDeviceToHost);	
        for(int i=0;i<n;i++)
           printf("%.2f ", temp[i]);
*/
  lga_filtering_forward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, input.data<float>(), filters.data<float>(),
			  height, width, channel, radius,
			  output.data<float>());
  //     temp = new float[n];


}


void lga_backward (at::Tensor input, at::Tensor filters, at::Tensor gradOutput,
	      at::Tensor gradInput, at::Tensor gradFilters, const int radius){

//      int num=input->size(0);
  int channel = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int n = filters.numel ();
  lga_filter_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, input.data<float>(),
			  gradOutput.data<float>(), height, width, channel,
			  radius, gradFilters.data<float>());
//    printf("%d, %d, %d, %d\n", height, width, channel, n);

  n = input.numel ();
  float *grad = gradInput.data<float>();
  cudaMemset (grad, 0, sizeof (float) * n);
  lga_data_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, filters.data<float>(),
			  gradOutput.data<float>(), height, width, channel,
			  radius, grad);

}

void lga3d_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
	       const int radius){

  //       int num=input->size(0);
  int channel = input.size(2);
  int height = input.size(3);
  int width = input.size(4);
  int n = input.numel ();
//        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  lga_filtering_forward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, input.data<float>(), filters.data<float>(),
			  height, width, channel, radius,
			  output.data<float>());

}


void lga3d_backward (at::Tensor input, at::Tensor filters, at::Tensor gradOutput,
		at::Tensor gradInput, at::Tensor gradFilters,
		const int radius){

//      int num=input->size(0);
  int channel = input.size(2);
  int height = input.size(3);
  int width = input.size(4);
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int n = filters.numel ();
  lga_filter_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, input.data<float>(),
			  gradOutput.data<float>(), height, width, channel,
			  radius, gradFilters.data<float>());

  n = input.numel ();
  float *grad = gradInput.data<float>();
  cudaMemset (grad, 0, sizeof (float) * n);
  lga_data_backward <<< (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
    CUDA_NUM_THREADS >>> (n, filters.data<float>(),
			  gradOutput.data<float>(), height, width, channel,
			  radius, grad);

}



#ifdef __cplusplus
    }
#endif
