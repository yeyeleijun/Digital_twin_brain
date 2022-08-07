#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

template<typename T>
static __global__ void update_extern_stimuli_kernel(const int* __restrict__ ext_stimulus_inds,
														const unsigned int n,
														const T* __restrict__ i_ext_inputs,
														T* __restrict__ i_ext_stimuli)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for (unsigned int i = idx; i < n; i += gridSize)
  	{
  		int index = ext_stimulus_inds[i];
		T ii = (index < 0) ? (T)0 : i_ext_inputs[index];
		i_ext_stimuli[i] = ii;
	}
}

template<typename T>
void update_extern_stimuli_gpu(const int* ext_stimulus_inds,
									const unsigned int n,
									const T* i_ext_inputs,
									T* i_ext_stimuli,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_extern_stimuli_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					ext_stimulus_inds,
					n,
					i_ext_inputs,
					i_ext_stimuli);
	HIP_POST_KERNEL_CHECK("update_membrane_voltage_kernel");
}

static __global__ void update_input_spike_kernel(const unsigned int* __restrict__ neuron_inds,
													const unsigned int n,
													unsigned char* __restrict__ f_actives)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for (unsigned int i = idx; i < n; i += gridSize)
  	{
	  	int fidx = neuron_inds[i];
		f_actives[fidx] = 0x01;
	}
}

void update_input_spike_gpu(const unsigned int* neuron_inds,
								const unsigned int n,
								unsigned char* f_actives,
								hipStream_t stream)
{
	hipLaunchKernelGGL(
					update_input_spike_kernel,
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					neuron_inds,
					n,
					f_actives);
	HIP_POST_KERNEL_CHECK("update_input_spike_kernel");
}


template void update_extern_stimuli_gpu<float>(const int* ext_stimulus_inds,
											const unsigned int n,
											const float* i_ext_inputs,
											float* i_ext_stimuli,
											hipStream_t stream);

}//namespace dtb
