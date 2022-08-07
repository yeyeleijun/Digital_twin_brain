#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

static __global__ void setup_noise_kernel(const unsigned long long seed,
											const unsigned long long seq,
											const unsigned long long offset,
											const unsigned int n,
											hiprandStatePhilox4_32_10_t *states)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence
	number,  offset */
	if(idx < n)
	{
		hiprand_init(seed, seq + idx * 4, offset, &states[idx]);
	}
}

void setup_noise_gpu(const unsigned long long seed,
					const unsigned long long seq,
					const unsigned long long offset,
					const unsigned int n,
					hiprandStatePhilox4_32_10_t *states,
					hipStream_t stream)
{
	hipLaunchKernelGGL(
					setup_noise_kernel,
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					seed,
					seq,
					offset,
					n,
					states);
	HIP_POST_KERNEL_CHECK("setup_noise_kernel");
}

template<typename T, typename T2>
static __global__ void init_noise_weight_kernel(const unsigned int* rowptrs,
											const unsigned int* colinds,
											const T2* w_synaptics,
											const unsigned char* connkinds,
											const unsigned int n,
											const int* noise_inds,
											T2* noise_ex_weights,
											T2* noise_in_weights)
{
	const unsigned int logNumOfWarp = power_radix2(warpSize);
	const unsigned int warpId = threadIdx.x >> logNumOfWarp;
	const unsigned int laneId = threadIdx.x & (warpSize - 1);
	const unsigned int warpsPerBlock = (blockDim.x >> logNumOfWarp);
	const unsigned int idx = blockIdx.x * warpsPerBlock + warpId;
	const unsigned int gridSize = warpsPerBlock * gridDim.x;
	unsigned int start, end;

	for(unsigned int i = idx; i < n; i += gridSize)
	{
		start = rowptrs[i];
		end = rowptrs[i + 1];

		for(unsigned int j = start + laneId; j < end; j += warpSize)
		{
			unsigned int nid = colinds[j];
			int nidx = noise_inds[nid];
			if(nidx >= 0)
			{
				T2 weight = w_synaptics[j];
				unsigned char flag = connkinds[j];
				if(flag)
				{
					atomic_add<T>(&(noise_in_weights[nidx].x), weight.x);
					atomic_add<T>(&(noise_in_weights[nidx].y), weight.y);
				}
				else
				{
					atomic_add<T>(&(noise_ex_weights[nidx].x), weight.x);
					atomic_add<T>(&(noise_ex_weights[nidx].y), weight.y);
				}
			}
		}
	}
}

template<typename T, typename T2>
void init_noise_weight_gpu(const unsigned int* rowptrs,
							const unsigned int* colinds,
							const T2* w_synaptics,
							const unsigned char* connkinds,
							const unsigned int n,
							const int* noise_inds,
							T2* noise_ex_weights,
							T2* noise_in_weights,
							hipStream_t stream)
{
	unsigned int warps_per_block = HIP_THREADS_PER_BLOCK >> (power_radix2(warp_size()));
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(init_noise_weight_kernel<T, T2>),
					dim3(divide_up<unsigned int>(n, warps_per_block)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					rowptrs,
					colinds,
					w_synaptics,
					connkinds,
					n,
					noise_inds,
					noise_ex_weights,
					noise_in_weights);
	HIP_POST_KERNEL_CHECK("init_noise_weight_kernel");
}


template void init_noise_weight_gpu<float, float2>(const unsigned int* rowptrs,
												const unsigned int* colinds,
												const float2* w_synaptics,
												const unsigned char* connkinds,
												const unsigned int n,
												const int* noise_inds,
												float2* noise_ex_weights,
												float2* noise_in_weights,
												hipStream_t stream);
}//namespace istbi 