#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

#define BLOCK_REDUCE_ASUM_TWO(TNUM) 			\
if (BlockSize >= (TNUM) * 2) { 					\
  if (tid < (TNUM)) { 							\
    tsum_replace(st1, s_sum1[tid + (TNUM)]); 	\
    tsum_replace(st2, s_sum2[tid + (TNUM)]); 	\
  } 											\
  __syncthreads(); 								\
}

#define REDUCE_ASUM_TOW(TNUM) 					\
if (tid + (TNUM) < thread_count) { 				\
  tsum_replace(st1, s_sum1[tid + (TNUM)]); 		\
  tsum_replace(st2, s_sum2[tid + (TNUM)]); 		\
  __syncthreads(); 								\
}


template<typename T, typename T2, unsigned int BlockSize>
__device__ void asum_shmem(volatile T *s_sum1, const T sum1, volatile T2 *s_sum2,
								const T2 sum2, unsigned int tid)
{
	const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
  	volatile T* st1 = s_sum1 + tid;
	volatile T2* st2 = s_sum2 + tid;
	*st1 = sum1;
	*st2 = sum2;
  	__syncthreads();
    // do reduction in shared mem
  	BLOCK_REDUCE_ASUM_TWO(256)
  	BLOCK_REDUCE_ASUM_TWO(128)
  	BLOCK_REDUCE_ASUM_TWO(64)
  	
    if (tid < 32)
    {
    	REDUCE_ASUM_TOW(32)
    	REDUCE_ASUM_TOW(16)
    	REDUCE_ASUM_TOW(8)
    	REDUCE_ASUM_TOW(4)
    	REDUCE_ASUM_TOW(2)
    	REDUCE_ASUM_TOW(1)
    }
}

#define BLOCK_REDUCE_ASUM(TNUM) 				\
if (BlockSize >= (TNUM) * 2) { 					\
  if (tid < (TNUM)) { 							\
    tsum_replace(st, s_sum[tid + (TNUM)]); 		\
  } 											\
  __syncthreads(); 								\
}

#define REDUCE_ASUM(TNUM) 						\
if (tid + (TNUM) < thread_count) { 				\
  tsum_replace(st, s_sum[tid + (TNUM)]); 		\
  __syncthreads(); 								\
}

template<typename T, unsigned int BlockSize>
__device__ void asum_shmem(volatile T *s_sum, const T sum, unsigned int tid)
{
	const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
  	volatile T* st = s_sum + tid;
	*st = sum;
  	__syncthreads();
    // do reduction in shared mem
  	BLOCK_REDUCE_ASUM(256)
  	BLOCK_REDUCE_ASUM(128)
  	BLOCK_REDUCE_ASUM(64)
  	
    if (tid < 32)
    {
    	REDUCE_ASUM(32)
    	REDUCE_ASUM(16)
    	REDUCE_ASUM(8)
    	REDUCE_ASUM(4)
    	REDUCE_ASUM(2)
    	REDUCE_ASUM(1)
    }
}

template<typename T, unsigned int blockSize>
__device__ void asum_block(const unsigned int pos,
							const unsigned int n,
							const unsigned int exclusive_count,
							const unsigned char* exclusive_flags,
							const T* v_membs,
							const T* i_synapes,
							T& v_sum,
							T& i_sum)
{
	__shared__ T s_sum1[blockSize];
	__shared__ T s_sum2[blockSize];
	
	T sum1 = static_cast<T>(0);
	T sum2 = static_cast<T>(0);
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		T vi = v_membs[idx];
		T ii = i_synapes[idx];
		sum1 += vi * (ei == 0x00 ? (T)1 : (T)0);
		sum2 += ii * (ei == 0x00 ? (T)1 : (T)0);
	}

	asum_shmem<T, T, blockSize>(s_sum1, sum1, s_sum2, sum2, threadIdx.x);

	if (threadIdx.x == 0)
	{
		v_sum = s_sum1[0] / (T)(n - exclusive_count);
		i_sum = s_sum2[0] / (T)(n - exclusive_count);
	}
}

template<unsigned int blockSize>
__device__ void asum_block(const unsigned int pos,
							const unsigned int n,
							const unsigned char* exclusive_flags,
							const unsigned char* f_actives,
							unsigned int& f_sum)
{
	__shared__ unsigned int s_sum[blockSize];
	
	unsigned int sum = 0;
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char fi = f_actives[idx];
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		sum += static_cast<unsigned int>(fi) * (ei == 0x00 ? 1 : 0);
	}

	asum_shmem<unsigned int, blockSize>(s_sum, sum, threadIdx.x);

	if (threadIdx.x == 0)
	{
		f_sum = s_sum[0];
	}
}

template<typename T, unsigned int blockSize>
__device__ void asum_block(const unsigned int pos,
							const unsigned int n,
							const unsigned int exclusive_count,
							const unsigned char* exclusive_flags,
							const T* in_data,
							T& out_sum)
{
	__shared__ T s_sum[blockSize];
	
	T sum = static_cast<T>(0);
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		T vi = in_data[idx] * (ei == 0x00 ? (T)1 : (T)0);
		sum += vi;
	}

	asum_shmem<T, blockSize>(s_sum, sum, threadIdx.x);

	if (threadIdx.x == 0)
	{
		out_sum = s_sum[0] / (n - exclusive_count);
	}
}

template<unsigned int BlockSize>
__global__ void stat_freqs_kernel(const uint2* sub_binfos,
									const unsigned int n,
									const unsigned char* exclusive_flags,
									const unsigned char* f_actives,
									unsigned int* freqs)
{
	uint2 info;
	unsigned int exclusive_count;
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		info = sub_binfos[i];
		
		if(info.y > 0)
		{
			asum_block<BlockSize>(info.x, info.y, exclusive_flags, f_actives, freqs[i]);
			__syncthreads();
		}
  	}
}

void stat_freqs_gpu(const uint2* sub_binfos,
					const unsigned int n,
					const unsigned char* exclusive_flags,
					const unsigned char* f_actives,
					unsigned int* freqs,
					hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_freqs_kernel<HIP_THREADS_PER_BLOCK>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_binfos,
					n,
					exclusive_flags,
					f_actives,
					freqs);
	HIP_POST_KERNEL_CHECK("stat_freqs_kernel");	
}


template<typename T, unsigned int BlockSize>
__global__ void stat_vmeans_and_imeans_kernel(const uint2* sub_binfos,
												const unsigned int* exclusive_counts,
												const unsigned int n,
												const unsigned char* exclusive_flags,
												const T* v_membranes,
												const T* i_synapses,
												T* vmeans,
												T* imeans)
{
	uint2 info;
	unsigned int exclusive_count;
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		info = sub_binfos[i];
		exclusive_count = (exclusive_counts == NULL) ? 0 : exclusive_counts[i];
		
		if(info.y > 0)
		{
			if(NULL != v_membranes && NULL != i_synapses)
				asum_block<T, BlockSize>(info.x, info.y, exclusive_count, exclusive_flags, v_membranes, i_synapses, vmeans[i], imeans[i]);
			else if(NULL != v_membranes)
				asum_block<T, BlockSize>(info.x, info.y, exclusive_count, exclusive_flags, v_membranes, vmeans[i]);
			else
				asum_block<T, BlockSize>(info.x, info.y, exclusive_count, exclusive_flags, i_synapses, imeans[i]);
			__syncthreads();
		}
  	}
}

template<typename T>
void stat_vmeans_and_imeans_gpu(const uint2* sub_binfos,
								const unsigned int* exclusive_counts,
								const unsigned int n,
								const unsigned char* exclusive_flags,
								const T* v_membranes,
								const T* i_synapses,
								T* vmeans,
								T* imeans,
								hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_vmeans_and_imeans_kernel<T, HIP_THREADS_PER_BLOCK>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_binfos,
					exclusive_counts,
					n,
					exclusive_flags,
					v_membranes,
					i_synapses,
					vmeans,
					imeans);
	HIP_POST_KERNEL_CHECK("stat_vmeans_and_imeans_kernel");	
}

template<typename T>
static __global__  void stat_spikes_and_vmembs_kernel(const unsigned int* samples,
															const unsigned int n,
															const unsigned char* f_actives,
															const T* v_membranes,
															char* spikes,
															T* vmembs)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = gridDim.x * blockDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int sid = samples[i];
		if(NULL != spikes)
		{
			spikes[i] = f_actives[sid];
		}
		
		if(NULL != vmembs)
			vmembs[i] = v_membranes[sid];
	}
}

template<typename T>
void stat_spikes_and_vmembs_gpu(const unsigned int* samples,
							const unsigned int n,
							const unsigned char* f_actives,
							const T* v_membranes,
							char* spikes,
							T* vmembs,
							hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_spikes_and_vmembs_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					samples,
					n,
					f_actives,
					v_membranes,
					spikes,
					vmembs);
	HIP_POST_KERNEL_CHECK("stat_spikes_and_vmembs_kernel");	
}

template void stat_vmeans_and_imeans_gpu<float>(const uint2* sub_binfos,
										const unsigned int* exclusive_counts,
										const unsigned int n,
										const unsigned char* exclusive_flags,
										const float* v_membranes,
										const float* i_synapses,
										float* vmeans,
										float* imeans,
										hipStream_t stream);

template void stat_spikes_and_vmembs_gpu<float>(const unsigned int* samples,
											const unsigned int n,
											const unsigned char* f_actives,
											const float* v_membranes,
											char* spikes,
											float* vmembs,
											hipStream_t stream);

}//namespace dtb
