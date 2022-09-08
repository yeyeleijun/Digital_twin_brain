#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

template<typename T, typename T2>
static __global__ void update_synaptic_current_kernel(const T2* j_ex_presynaptics,
											const T2* j_in_presynaptics,
											const T2* g_ex_conducts,
											const T2* g_in_conducts,
											const T2* v_ex_membranes,
											const T2* v_in_membranes,
											const T* v_membranes,
											const unsigned int n,
											T* i_synaptics)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
 	const unsigned int gridSize = blockDim.x * gridDim.x;
	
  	for(unsigned int i = idx; i < n; i += gridSize)
  	{
  		//Kahan summation algorithm
  		const T vi = v_membranes[i];
		
		const T2 vi_ex = v_ex_membranes[i];
		const T2 ji_ex = j_ex_presynaptics[i];
		const T2 gi_ex = g_ex_conducts[i];

		const T2 vi_in = v_in_membranes[i];
		const T2 ji_in = j_in_presynaptics[i];
		const T2 gi_in = g_in_conducts[i];


		T sum = gi_ex.x * (vi_ex.x - vi) * ji_ex.x;
#if 1
		sum += gi_ex.y * (vi_ex.y - vi) * ji_ex.y;
		sum += gi_in.x * (vi_in.x - vi) * ji_in.x;
		sum += gi_in.y * (vi_in.y - vi) * ji_in.y;
#else
		T y = gi_ex.y * (vi_ex.y - vi) * ji_ex.y;
		T t = sum + y;
		T c = (t - sum) - y;
		sum = t;

		y = gi_in.x * (vi_in.x - vi) * ji_in.x - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;

		y = gi_in.y * (vi_in.y - vi) * ji_in.y - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t - c;
#endif	
		i_synaptics[i] = sum;
	}
}

template<typename T, typename T2>
void update_synaptic_current_gpu(const T2* j_ex_presynaptics,
									const T2* j_in_presynaptics,
									const T2* g_ex_conducts,
									const T2* g_in_conducts,
									const T2* v_ex_membranes,
									const T2* v_in_membranes,
									const T* v_membranes,
									const unsigned int n,
									T* i_synaptics,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_synaptic_current_kernel<T, T2>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					j_ex_presynaptics,
					j_in_presynaptics,
					g_ex_conducts,
					g_in_conducts,
					v_ex_membranes,
					v_in_membranes,
					v_membranes,
					n,
					i_synaptics);
	HIP_POST_KERNEL_CHECK("update_synaptic_current_kernel");
}

template void update_synaptic_current_gpu<float, float2>(const float2* j_ex_presynaptics,
														const float2* j_in_presynaptics,
														const float2* g_ex_conducts,
														const float2* g_in_conducts,
														const float2* v_ex_membranes,
														const float2* v_in_membranes,
														const float* v_membranes,
														const unsigned int n,
														float* i_synaptics,
														hipStream_t stream);

}//namespace dtb 
