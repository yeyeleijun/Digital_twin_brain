#include <cassert>
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include "common.hpp"
#include "common.cuh"
#include "data_allocator.hpp"
#include "test/save_result.hpp"
#include "util/transpose.hpp"

using namespace std;

namespace istbi {

template<unsigned int warpsPerBlock>
static __global__  void save_exchange_spike_kernel(const unsigned char* f_actives,
											const unsigned int* offsets,
											const unsigned int* rowptrs,
											const unsigned int n,
											unsigned char* f_saving_actives)
{
	const unsigned int warpId = threadIdx.x / warpSize;
	const unsigned int laneId = threadIdx.x & (warpSize - 1);
	const unsigned int gridSize = warpsPerBlock * gridDim.x;
	unsigned int start, end;

	for(unsigned int i = blockIdx.x * warpsPerBlock + warpId; i < n; i += gridSize)
	{
		start = rowptrs[i];
		end = rowptrs[i + 1];
		unsigned int offset = (offsets[i] >> 2);
		for(unsigned int j = start + laneId; j < end; j += warpSize, offset++)
		{
			unsigned int fi = __ldg(reinterpret_cast<const unsigned int*>(f_actives) + offset);
			fi = __bfe(fi, laneId, 1);
			f_saving_actives[j] = static_cast<unsigned char>(fi);
		}
	}
}

void save_exchange_spike_gpu(const unsigned char* f_actives,
						const unsigned int* offsets,
						const unsigned int* rowptrs,
						const unsigned int n,
						unsigned char* f_saving_actives,
						cudaStream_t stream)
{
	unsigned int blocks = (n - 1) / (CUDA_THREADS_PER_BLOCK >> 5) + 1;
	save_exchange_spike_kernel<(CUDA_THREADS_PER_BLOCK >> 5)><<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(f_actives,
																									offsets,
																									rowptrs,
																									n,
																									f_saving_actives);
	CUDA_POST_KERNEL_CHECK("save_exchange_spike_kernel");
	
}

void save_exchange_spike_cpu(const unsigned char* f_actives,
						const unsigned int* offsets,
						const unsigned int* rowptrs,
						const unsigned int n,
						unsigned char* f_saving_actives)
{
	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int count = 0;
		unsigned int offset = offsets[i];
		for(unsigned int j = rowptrs[i]; j < rowptrs[i + 1]; j++, count++)
		{
			unsigned char flag = f_actives[offset + (count >> 3)];
			f_saving_actives[j] = ((flag >> (count & 7)) & 0x1);
		}
	}
}

template<unsigned int warpsPerBlock>
static __global__  void saving_sending_spike_kernel(const unsigned char* f_actives,
													const unsigned int* rowptrs,
													const unsigned int n,
													const unsigned int* colinds,
													unsigned char* f_saving_actives)
{
	const unsigned int warpId = threadIdx.x / warpSize;
	const unsigned int laneId = threadIdx.x & (warpSize - 1);
	const unsigned int gridSize = warpsPerBlock * gridDim.x;
	unsigned int start, end;

	for(unsigned int i = blockIdx.x * warpsPerBlock + warpId; i < n; i += gridSize)
	{
		start = rowptrs[i];
		end = rowptrs[i + 1];
		for(unsigned int j = start + laneId; j < end; j += warpSize)
		{
			unsigned int fidx = __ldg(colinds + j);
			unsigned int fi = __ldg(reinterpret_cast<const unsigned int*>(f_actives) + (fidx >> 5));
			fi = __bfe(fi, (fidx & 31), 1);
			f_saving_actives[j] = static_cast<unsigned char>(fi);
		}
	}
}

void saving_sending_spike_gpu(const unsigned char* f_actives,
									const unsigned int* rowptrs,
									const unsigned int n,
									const unsigned int* colinds,
									unsigned char* f_saving_actives,
									cudaStream_t stream)
{
	unsigned int blocks = (n - 1) / (CUDA_THREADS_PER_BLOCK >> 5) + 1;
	saving_sending_spike_kernel<(CUDA_THREADS_PER_BLOCK >> 5)><<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(f_actives,
																											rowptrs,
																											n,
																											colinds,
																											f_saving_actives);
	CUDA_POST_KERNEL_CHECK("saving_sending_spike_kernel");
	
}

static __global__  void save_spike_kernel(const unsigned char* f_actives,
											const unsigned int n,
											unsigned char* f_saving_actives)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int laneId = threadIdx.x & (warpSize - 1);
	const unsigned int gridSize = blockDim.x * gridDim.x;

	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int fi = __ldg(reinterpret_cast<const unsigned int*>(f_actives) + (i >> 5));
		fi = __bfe(fi, laneId, 1);
		f_saving_actives[i] = static_cast<unsigned char>(fi);
	}
}

void save_spike_gpu(const unsigned char* f_actives,
						const unsigned int n,
						unsigned char* f_saving_actives,
						cudaStream_t stream)
{
	unsigned int blocks = (n - 1) / CUDA_THREADS_PER_BLOCK + 1;
	save_spike_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(f_actives,
																	n,
																	f_saving_actives);
	CUDA_POST_KERNEL_CHECK("save_spike_kernel");
	
}

template<typename T>
void resort_result_gpu(const long long* d_resort_inds,
						const T* d_keys,
						const unsigned int n,
						T* d_vals)
{
	thrust::device_ptr<long long> d_maps = thrust::device_pointer_cast(const_cast<long long*>(d_resort_inds));
	thrust::gather(d_maps, d_maps + n, thrust::device_pointer_cast(d_keys), thrust::device_pointer_cast(d_vals));
}

template<typename T>
void resort_params_cpu(unsigned long long* nids,
						unsigned int n,
						unsigned char* kinds,
						T* weights)
{
	thrust::host_vector<unsigned long long> keys(nids, nids + n);
	if(!thrust::is_sorted(keys.begin(), keys.end()))
	{
		thrust::host_vector<unsigned int> maps(n);
		thrust::sequence(maps.begin(), maps.end());
		thrust::sort_by_key(keys.begin(), keys.end(), maps.begin());
		thrust::copy(keys.begin(), keys.end(), nids);
		{
			thrust::host_vector<unsigned char> vals(kinds, kinds + n);
			thrust::gather(maps.begin(), maps.end(), vals.begin(), kinds);
		}

		{
			thrust::host_vector<T> vals(weights, weights + n);
			thrust::gather(maps.begin(), maps.end(), vals.begin(), weights);
		}
	}
}

void resort_samples_gpu(long long* samples,
							const unsigned int height,
							const unsigned int width,
							const unsigned int bid,
							vector<unsigned int>& indices)
{
	static unsigned int upper_size = 65535 * 32;
	assert(width == 2);
	thrust::device_vector<unsigned int> d_keys(height);
	thrust::device_vector<unsigned int> d_vals(height);
	
	{
		std::size_t total = static_cast<std::size_t>(height) * width;
		thrust::device_vector<long long> d_samples(total);
		
		if(height <= upper_size)
		{
			thrust::device_vector<long long> d_temps(total);
			thrust::copy(samples, samples + total, d_temps.begin());

			transpose_gpu<long long>(thrust::raw_pointer_cast(d_temps.data()),
									height,
									width,
									thrust::raw_pointer_cast(d_samples.data()));
			CUDA_CHECK(cudaDeviceSynchronize());			
		}
		else
		{
			thrust::host_vector<long long> h_temps(total);
			transpose_cpu<long long>(samples, height, width, h_temps.data());
			thrust::copy(h_temps.begin(), h_temps.end(), d_samples.begin());
		}

		thrust::identity<long long> op;
		thrust::transform(d_samples.begin(), d_samples.begin() + height, d_keys.begin(), op);
		thrust::transform(d_samples.begin() + height, d_samples.end(), d_vals.begin(), op);
	}

	if(!thrust::is_sorted(d_keys.begin(), d_keys.end()))
	{
		thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());
	}

	thrust::device_vector<unsigned int>::iterator it_beg = thrust::lower_bound(d_keys.begin(), d_keys.end(), bid);
	if(it_beg != d_keys.end())
	{
		thrust::device_vector<unsigned int>::iterator it_end = thrust::upper_bound(d_keys.begin(), d_keys.end(), bid);
		unsigned int beg = it_beg - d_keys.begin();
		unsigned int end = it_end - d_keys.begin();
		unsigned int count = end - beg;
		if(count > 0)
		{
			indices.resize(count);
			if(!thrust::is_sorted(d_vals.begin() + beg, d_vals.begin() + end))
			{
				thrust::sort(d_vals.begin() + beg, d_vals.begin() + end);
			}
			thrust::copy(d_vals.begin() + beg, d_vals.begin() + end, indices.data());
		}
	}
}

void resort_samples_cpu(long long* samples,
							const unsigned int height,
							const unsigned int width,
							const unsigned int bid,
							vector<unsigned int>& indices)
{
	assert(width == 2);
	for(unsigned int i = 0; i < height; i++)
	{
		if(samples[i * width] == bid)
		{
			indices.push_back(samples[i * width + 1]);
		}
	}

	if(!thrust::is_sorted(indices.data(), indices.data() + indices.size()))
	{
		thrust::sort(indices.data(), indices.data() + indices.size());
	}
}



template void resort_result_gpu<unsigned char>(const long long* h_resort_inds,
					const unsigned char* d_keys,
					const unsigned int n,
					unsigned char* d_vals);

template void resort_result_gpu<float>(const long long* h_resort_inds,
						const float* d_keys,
						const unsigned int n,
						float* d_vals);

template void resort_result_gpu<double>(const long long* h_resort_inds,
					const double* d_keys,
					const unsigned int n,
					double* d_vals);

template void resort_result_gpu<float2>(const long long* h_resort_inds,
					const float2* d_keys,
					const unsigned int n,
					float2* d_vals);

template void resort_result_gpu<double2>(const long long* h_resort_inds,
						const double2* d_keys,
						const unsigned int n,
						double2* d_vals);

template void resort_params_cpu<float2>(unsigned long long* nids,
						unsigned int n,
						unsigned char* kinds,
						float2* weights);

template void resort_params_cpu<double2>(unsigned long long* nids,
						unsigned int n,
						unsigned char* kinds,
						double2* weights);

}//namespace istbi
