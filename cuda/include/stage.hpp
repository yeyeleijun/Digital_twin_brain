#pragma once
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include <hiprand_kernel.h>

using namespace std;

namespace dtb {

#define MTGP32_MAX_NUM_BLOCKS 200

template<typename T>
void update_extern_stimuli_gpu(const int* ext_stimulus_inds,
									const unsigned int n,
									const T* i_ext_inputs,
									T* i_ext_stimuli,
									hipStream_t stream = NULL);

void update_input_spike_gpu(const unsigned int* neuron_inds,
								const unsigned int n,
								unsigned char* f_actives,
								hipStream_t stream = NULL);

template<typename T>
void init_membrane_voltage_gpu(const T* v_ths,
									const T* v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream = NULL);

template<typename T>
void reset_membrane_voltage_gpu(const T*  v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream = NULL);

template<typename T>
void update_membrane_voltage_for_input_gpu(const T* v_rsts,
													const T* v_ths,
													const unsigned char* f_actives,
													const unsigned int n,
													T* v_membranes,
													hipStream_t stream = NULL);

template<typename T>
void update_membrane_voltage_gpu(const T* i_synaptics,
										const T* i_ext_stimuli,
										const T* v_rsts,
										const T* v_ths,
										const T* c_membranes,
										const T* v_leakages,
										const T* g_leakages,
										const T* t_refs,
										const unsigned int n,
										const T delta_t,
										const T t,
										unsigned char* f_actives,
										T* t_actives,
										T* v_membranes,
										hipStream_t stream = NULL);

template<typename T>
void init_spike_time_gpu(const unsigned int n,
							const T val,
							T* t_actives,
							hipStream_t stream = NULL);

							
void create_generator_state(const unsigned long long seed, 
							hiprandStateMtgp32* states,
							mtgp32_kernel_params_t* params);

template<typename T>
void generate_uniform_samples(hiprandStateMtgp32* states,
							const unsigned int n,
							const T a,
							const T b,
							T* samples,
							hipStream_t stream = NULL);

template<typename T>
void update_spike_gpu(hiprandStateMtgp32* states,
						const unsigned int n,
						const T* noise_rates,
						unsigned char* f_actives,
						const T a = 0.f,
						const T b = 1.f,
						T* samples = NULL,
						hipStream_t stream = NULL);

void update_routing_offsets_gpu(const unsigned int* routing_unions,
									const unsigned int routing_union_nums,
									const unsigned int routing_offset_nums,
									unsigned int* routing_offsets,
									hipStream_t stream = NULL);

void update_routing_neuron_ids_gpu(const unsigned int* inputs,
									const size_t size,
									unsigned int* outputs,	
									hipStream_t stream = NULL);

void update_recving_spikes_gpu(const unsigned int* inputs,
									const unsigned int* rowptrs,
									const unsigned int  segments,
									unsigned int* outputs,	
									hipStream_t stream = NULL);

void update_routing_spikes_gpu(const unsigned int* inputs,
								const size_t size,
								unsigned char* outputs,
								hipStream_t stream = NULL);

void count_sending_spikes_temporary_storage_size(const unsigned int sending_count,
														const unsigned int segments,
														unsigned int* block_rowptrs,
														unsigned int* active_rowptrs,
														size_t& storage_size_bytes,
														hipStream_t stream = NULL);

void update_sending_spikes_gpu(const unsigned char* f_actives,
									const unsigned int* sending_rowptrs,
									const unsigned int* sending_colinds,
									const unsigned int segments_count,
									const unsigned int sending_count,
									unsigned char* f_sending_actives,
									unsigned int* block_rowptrs,
									unsigned int* active_rowptrs,
									unsigned int* active_colinds,
									hipStream_t stream = NULL);


void setup_noise_gpu(const unsigned long long seed,
				const unsigned long long seq,
				const unsigned long long offset,
				const unsigned int n,
				hiprandStatePhilox4_32_10_t *states,
				hipStream_t stream = NULL);

template<typename T, typename T2>
void init_noise_weight_gpu(const unsigned int* rowptrs,
							const unsigned int* colinds,
							const T2* w_synaptics,
							const unsigned char* connkinds,
							const unsigned int n,
							const int* noise_inds,
							T2* noise_ex_weights,
							T2* noise_in_weights,
							hipStream_t stream = NULL);

template<typename T, typename T2>
void transform_presynaptic_voltage_gpu(const T2* j_ex_presynaptics,
										const T2* j_in_presynaptics,
										const unsigned int n,
										T* j_presynaptics,
										hipStream_t stream = NULL);

template<typename T2>
void init_presynaptic_voltage_gpu(const unsigned int n,
									const T2 val,
									T2* j_u_presynaptics,
									hipStream_t stream = NULL);


template<typename T, typename T2> 
void update_presynaptic_voltage_gpu(const T2* tao_ex_constants,
												const T2* tao_in_constants,
												const unsigned int n,
												T2* j_ex_presynaptics,
												T2* j_ex_presynaptic_deltas,
												T2* j_in_presynaptics,
												T2* j_in_presynaptic_deltas,
												hipStream_t stream = NULL);

template<typename T, typename T2>
void update_presynaptic_voltage_inner_gpu(const unsigned int* rowptrs,
										const unsigned int* colinds,
										const T2* w_synaptics,
										const unsigned char* connkinds,
										const unsigned int n,
										const unsigned int* f_indices,
										const unsigned char* f_actives,
										T2* j_ex_presynaptics,
										T2* j_in_presynaptics,
										hipStream_t stream = NULL);

template<typename T, typename T2>
void update_presynaptic_voltage_outer_gpu(const unsigned int* rowptrs,
										const unsigned int* colinds,
										const T2* w_synaptics,
										const unsigned char* connkinds,
										const unsigned int* active_colinds,
										const unsigned int n,
										T2* j_ex_presynaptics,
										T2* j_in_presynaptics,
										hipStream_t stream = NULL);

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
									hipStream_t stream = NULL);

void stat_freqs_gpu(const uint2* sub_binfos,
					const unsigned int n,
					const unsigned char* exclusive_flags,
					const unsigned char* f_actives,
					unsigned int* freqs,
					hipStream_t stream = NULL);

template<typename T>
void stat_vmeans_and_imeans_gpu(const uint2* sub_binfos,
								const unsigned int* exclusive_counts,
								const unsigned int n,
								const unsigned char* exclusive_flags,
								const T* v_membranes,
								const T* i_synapses,
								T* vmeans,
								T* imeans,
								hipStream_t stream = NULL);

template<typename T>
void stat_spikes_and_vmembs_gpu(const unsigned int* samples,
										const unsigned int n,
										const unsigned char* f_actives,
										const T* v_membranes,
										char* spikes,
										T* vmembs,
										hipStream_t stream = NULL);

template<typename T>
void update_sample_neuron_gpu(const unsigned char* f_actives,
								const T* v_membranes,
								const unsigned int* indices,
								const unsigned int n,
								unsigned char* f_sample_actives,
								T* v_sample_membranes,
								hipStream_t stream = NULL);

enum PropType{
	EXT_STIMULI_I = 0,
	MEMBRANE_C = 1,
	REF_T = 2,
	LEAKAGE_G = 3,
	LEAKAGE_V = 4,
	THRESHOLD_V = 5,
	RESET_V = 6,
	CONDUCT_G_AMPA = 7,
	CONDUCT_G_NMDA = 8,
	CONDUCT_G_GABAa = 9,
	CONDUCT_G_GABAb = 10,
	MEMBRANE_V_AMPA = 11,
	MEMBRANE_V_NMDA = 12,
	MEMBRANE_V_GABAa = 13,
	MEMBRANE_V_GABAb = 14,
	TAO_AMPA = 15,
	TAO_NMDA = 16,
	TAO_GABAa = 17,
	TAO_GABAb = 18,
	NOISE_RATE = 19
};

template<typename T, typename T2>
struct Properties
{
	T* noise_rates;
	T* i_ext_stimuli;
	T* c_membrane_reciprocals;
	T* t_refs;
	T* g_leakages;
	T* v_leakages;
	T* v_thresholds;
	T* v_resets;
	T2* g_ex_conducts;
	T2* g_in_conducts;
	T2* v_ex_membranes;
	T2* v_in_membranes;
	T2* tao_ex_constants;
	T2* tao_in_constants;
	unsigned int n;
};

template<typename T, typename T2>
void update_props_gpu(const unsigned int* neuron_indice,
						const unsigned int* prop_indice,
						const T* prop_vals,
						const unsigned int n,
						Properties<T, T2>& prop,
						hipStream_t stream = NULL);

template<typename T, typename T2>
void update_prop_cols_gpu(const unsigned int* sub_bids,
							const uint2* sub_binfos,
							const unsigned int m,
							const unsigned int* prop_indice,
							const unsigned int* brain_indice,
							const T* hp_vals,
							const unsigned int n,
							Properties<T, T2>& prop,
							hipStream_t stream = NULL);

template<typename T, typename T2>
void gamma_gpu(const unsigned int* sub_bids,
					const uint2* sub_binfos,
					const unsigned int m,
					const unsigned int* prop_indice,
					const unsigned int* brain_indice,
					const T* alpha,
					const T* beta,
					const unsigned int n,
					const unsigned long long seed,
					const unsigned long long offset,
					Properties<T, T2>& prop,
					hipStream_t stream = NULL);

template<typename T>
void gamma_gpu(const unsigned int prop_id,
					const T alpha,
					const T beta,
					const unsigned long long seed,
					const unsigned long long offset,
					const T min_val,
					const unsigned int x,
					const unsigned int n,
					T* gamma,
					hipStream_t stream = NULL);

template<typename T>
void gamma_gpu(const unsigned int prop_id,
					const T alpha,
					const T beta,
					const unsigned long long seed,
					const unsigned long long offset,
					const T min_val,
					const T max_val,
					const unsigned int x,
					const unsigned int n,
					T* gamma,
					hipStream_t stream = NULL);
}//namespace istbi 