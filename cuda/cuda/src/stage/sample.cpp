#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

template<typename T>
static __global__  void update_sample_neuron_kernel(const unsigned char* f_actives,
											const T* v_membranes,
											const unsigned int* sample_indices,
											const unsigned int n,
											unsigned char* f_sample_actives,
											T* v_sample_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = gridDim.x * blockDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		const unsigned int fidx = sample_indices[i];
		const unsigned char fi = f_actives[fidx];
		const T vi = v_membranes[fidx];
		f_sample_actives[i] = fi;
		v_sample_membranes[i] = vi;
	}
}

template<typename T>
void update_sample_neuron_gpu(const unsigned char* f_actives,
								const T* v_membranes,
								const unsigned int* sample_indices,
								const unsigned int n,
								unsigned char* f_sample_actives,
								T* v_sample_membranes,
								hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_sample_neuron_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					f_actives,
					v_membranes,
					sample_indices,
					n,
					f_sample_actives,
					v_sample_membranes);
	HIP_POST_KERNEL_CHECK("update_sample_neuron_kernel");
	
}

template void update_sample_neuron_gpu<float>(const unsigned char* f_actives,
											const float* v_membranes,
											const unsigned int* indices,
											const unsigned int n,
											unsigned char* f_sample_actives,
											float* v_sample_membranes,
											hipStream_t stream);

}//namespace dtb

