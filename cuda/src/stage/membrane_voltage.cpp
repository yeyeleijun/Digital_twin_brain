#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

template<typename T>
static __global__ void init_membrane_voltage_kernel(const T* v_ths,
														const T* v_rsts,
														const unsigned int n,
														T* v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
		v_membranes[i] = (v_ths[i] + v_rsts[i]) / 2;
	}
}


template<typename T>
void init_membrane_voltage_gpu(const T* v_ths,
									const T* v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(init_membrane_voltage_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_ths,
					v_rsts,
					n,
					v_membranes);
	HIP_POST_KERNEL_CHECK("init_membrane_voltage_kernel");
}

template<typename T>
static __global__ void reset_membrane_voltage_kernel(const T*  v_rsts,
													const unsigned int n,
													T* v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
		v_membranes[i] = v_rsts[i];
	}
}

template<typename T>
void reset_membrane_voltage_gpu(const T*  v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(reset_membrane_voltage_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_rsts,
					n,
					v_membranes);
	HIP_POST_KERNEL_CHECK("reset_membrane_voltage_kernel");
}

template<typename T>
static __global__ void update_membrane_voltage_for_input_kernel(const T* __restrict__ v_rsts,
																		const T* __restrict__ v_ths,
																		const unsigned char* __restrict__ f_actives,
																		const unsigned int n,
																		T* __restrict__ v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for (unsigned int i = idx; i < n; i += gridSize)
  	{
		unsigned char fi = f_actives[i];
		v_membranes[i] = fi ? v_ths[i] : v_rsts[i];
	}
}

template<typename T>
void update_membrane_voltage_for_input_gpu(const T* v_rsts,
										const T* v_ths,
										const unsigned char* f_actives,
										const unsigned int n,
										T* v_membranes,
										hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_membrane_voltage_for_input_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_rsts,
					v_ths,
					f_actives,
					n,
					v_membranes);
	HIP_POST_KERNEL_CHECK("update_membrane_voltage_for_input_kernel");
}



template<typename T>
static __global__ void update_membrane_voltage_kernel(const T* __restrict__ i_synaptics,
															const T* __restrict__ i_ext_stimuli,
															const T* __restrict__ v_rsts,
															const T* __restrict__ v_ths,
															const T* __restrict__ c_membrane_reciprocals,
															const T* __restrict__ v_leakages,
															const T* __restrict__ g_leakages,
															const T* __restrict__ t_refs,
															const unsigned int n,
															const T delta_t,
															const T t,
															unsigned char* __restrict__ f_actives,
															T* __restrict__ t_actives,
															T* __restrict__ v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	T vi;
	
	for (unsigned int i = idx; i < n; i += gridSize)
  	{	
  		unsigned char fchanged = 0x0;
		unsigned char vchanged = 0x0;
		
		unsigned char fi = f_actives[i];
		if(fi)
		{
			vi = v_rsts[i];
			vchanged = 0x1;
			fchanged = 0x1;
			fi = 0x0;
		}
		
		if(t >= (t_actives[i] + t_refs[i]))
		{
			vi = v_membranes[i];
			T vth = v_ths[i];
			T cvi = g_leakages[i] * (v_leakages[i] - vi) + i_synaptics[i] + i_ext_stimuli[i];
			vi += delta_t * c_membrane_reciprocals[i] * cvi;
			
			if(vi >= vth)
			{
				vi = vth;
				fi = 0x01;
				fchanged = 0x1;
			}

			vchanged = 0x1;
		}

		if(fi)
			t_actives[i] = t;

		if(fchanged)
			f_actives[i] = fi;	
		
		if(vchanged)
			v_membranes[i] = vi;
		
	}
}

template<typename T>
void update_membrane_voltage_gpu(const T* i_synaptics,
										const T* i_ext_stimuli,
										const T* v_rsts,
										const T* v_ths,
										const T* c_membrane_reciprocals,
										const T* v_leakages,
										const T* g_leakages,
										const T* t_refs,
										const unsigned int n,
										const T delta_t,
										const T t,
										unsigned char* f_actives,
										T* t_actives,
										T* v_membranes,
										hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_membrane_voltage_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					i_synaptics,
					i_ext_stimuli,
					v_rsts,
					v_ths,
					c_membrane_reciprocals,
					v_leakages,
					g_leakages,
					t_refs,
					n,
					delta_t,
					t,
					f_actives,
					t_actives,
					v_membranes);
	HIP_POST_KERNEL_CHECK("update_membrane_voltage_kernel");
}


template void init_membrane_voltage_gpu<float>(const float* v_ths,
											const float* v_rsts,
											const unsigned int n,
											float* v_membranes,
											hipStream_t stream);

template void reset_membrane_voltage_gpu<float>(const float*  v_rsts,
											const unsigned int n,
											float* v_membranes,
											hipStream_t stream);

template void update_membrane_voltage_for_input_gpu<float>(const float* v_rsts,
														const float* v_ths,
														const unsigned char* f_actives,
														const unsigned int n,
														float* v_membranes,
														hipStream_t stream);


template void update_membrane_voltage_gpu<float>(const float* i_synaptics,
												const float* i_ext_stimuli,
												const float* v_rsts,
												const float* v_ths,
												const float* c_membrane_reciprocals,
												const float* v_leakages,
												const float* g_leakages,
												const float* t_refs,
												const unsigned int n,
												const float delta_t,
												const float t,
												unsigned char* f_actives,
												float* t_actives,
												float* v_membranes,
												hipStream_t stream);

}//namespace dtb
