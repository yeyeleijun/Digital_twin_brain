#include <cassert>
#include <iostream>
#include <thrust/find.h>
#include <thrust/binary_search.h>
#include "block.hpp"
#include "common.hpp"
#include "stage.hpp"
#include "configuration.hpp"

namespace dtb {

template<typename T, typename T2>
BrainBlock<T, T2>::BrainBlock(const unsigned short block_id,
								const int gpu_id,
								const T delta_t,
								const T t,
								const unsigned long long seed)
	:block_type_(BLOCK_TYPE_NORMAL),
	bid_(block_id),
	gpu_id_(gpu_id),
	seed_(seed),
	delta_t_(delta_t),
	t_(t),
	philox_seed_offset_(0),
	input_timestamps_(nullptr),
	input_rowptrs_(nullptr),
	input_colinds_(nullptr)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{		
		gen_states_ = make_shared<DataAllocator<hiprandStateMtgp32>>(static_cast<int>(bid_ + 1), MTGP32_MAX_NUM_BLOCKS * sizeof(hiprandStateMtgp32));
		kernel_params_ = make_shared<DataAllocator<mtgp32_kernel_params_t>>(static_cast<int>(bid_ + 1), sizeof(mtgp32_kernel_params_t));
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::init_config_params_gpu(const std::string& filename)
{
	ConfigParameter<T, T2> params;
	parse_params_from_numpy<T, T2, unsigned short>(bid_,
							filename,
							delta_t_,
							total_neurons_,
							params);

	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		j_ex_presynaptics_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_ex_presynaptic_deltas_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_ex_presynaptic_deltas_->gpu_data();
		j_in_presynaptics_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_in_presynaptic_deltas_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_in_presynaptic_deltas_->gpu_data();
		i_synaptics_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
		t_actives_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
		uniform_samples_ = nullptr;
	}
	v_membranes_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
	f_inner_actives_ = make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), total_neurons_);
	f_inner_actives_without_noise_ = make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), total_neurons_, false);
	
	f_exclusive_flags_ = params.exclusive_flags;
	f_exclusive_counts_ = params.exclusive_counts;
	
	sub_bids_ = params.subids;
	sub_bcounts_ = params.subcounts;
	sub_binfos_ = params.subinfos;

	if(nullptr != sub_binfos_)
	{
		if(nullptr != freqs_)
			freqs_.reset();
		freqs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), sub_binfos_->count() * sizeof(unsigned int));

		if(nullptr != vmeans_)
			vmeans_.reset();
		vmeans_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), sub_binfos_->count() * sizeof(T));

		if(nullptr != imeans_)
			imeans_.reset();
		imeans_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), sub_binfos_->count() * sizeof(T));
	}

	v_resets_ = params.v_resets;
	v_thresholds_ = params.v_thresholds;
		
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		noise_rates_ = params.noise_rates;
		i_ext_stimuli_= params.i_ext_stimuli;
		g_ex_conducts_ = params.g_ex_conducts;
		g_in_conducts_ = params.g_in_conducts;
		v_ex_membranes_ = params.v_ex_membranes;
		v_in_membranes_ = params.v_in_membranes;
		tao_ex_constants_ = params.tao_ex_constants;
		tao_in_constants_ = params.tao_in_constants;
		c_membrane_reciprocals_ = params.c_membrane_reciprocals;
		v_leakages_ = params.v_leakages;
		g_leakages_ = params.g_leakages;
		t_refs_ = params.t_refs;
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::init_connection_table_gpu(const std::string& filename)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));

	InputSpike input;
	ConnectionTable<unsigned short, T2> tab;
	if(CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_INPUT == parse_conn_table_from_numpy<T, T2, unsigned short>(
													bid_,
													filename,
													input,
													tab))
	{
		block_type_ = BLOCK_TYPE_INPUT;
		input_timestamps_ = input.input_timestamps;
		input_rowptrs_ = input.input_rowptrs;
		input_colinds_ = input.input_colinds;
		return;
	}

	block_type_ = BLOCK_TYPE_NORMAL;
	inner_conninds_ = tab.inner_conninds;
	
	inner_rowptrs_ = tab.inner_rowptrs;
	inner_colinds_ = tab.inner_colinds;
	inner_w_synaptics_ = tab.inner_vals;
	inner_connkinds_ = tab.inner_connkinds;
	
	f_receiving_bids_ = tab.outer_conn_bids;
	f_receiving_rowptrs_ = tab.outer_conn_inds;
	f_receiving_colinds_ = tab.outer_conn_nids;

	outer_rowptrs_ = tab.outer_rowptrs;
	outer_colinds_ = tab.outer_colinds;
	outer_w_synaptics_ = tab.outer_vals;
	outer_connkinds_ = tab.outer_connkinds;
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::init_all_stages_gpu()
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		init_membrane_voltage_gpu<T>(v_thresholds_->gpu_data(),
							v_resets_->gpu_data(),
							total_neurons_,
							v_membranes_->mutable_gpu_data());

		create_generator_state(seed_, gen_states_->mutable_gpu_data(), kernel_params_->mutable_gpu_data());
		init_spike_time_gpu<T>(total_neurons_, 0.f, t_actives_->mutable_gpu_data());

		if(!f_receiving_bids_.empty())
		{
			HIP_CHECK(hipMemcpy(f_receiving_rowptrs_->mutable_gpu_data(), f_receiving_rowptrs_->cpu_data(), f_receiving_rowptrs_->size(), hipMemcpyHostToDevice));
			f_receiving_active_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), f_receiving_rowptrs_->size());
			f_receiving_active_rowptrs_->cpu_data();
			f_receiving_active_rowptrs_->gpu_data();
			f_receiving_active_colinds_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), f_receiving_colinds_->size());
			f_receiving_active_colinds_->cpu_data();
			f_receiving_active_colinds_->gpu_data();
		}
		
		T2 val = {0.f, 0.f};
		init_presynaptic_voltage_gpu<T2>(total_neurons_,
										val,
										j_ex_presynaptics_->mutable_gpu_data());
		
		init_presynaptic_voltage_gpu<T2>(total_neurons_,
										val,
										j_in_presynaptics_->mutable_gpu_data());
		update_I_synaptic_gpu();
	}
	
	HIP_CHECK(hipMemset(f_inner_actives_->mutable_gpu_data(), 0x00, f_inner_actives_->size()));
}

template<typename T, typename T2>
void BrainBlock<T, T2>::reset_V_membrane_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		reset_membrane_voltage_gpu<T>(v_resets_->gpu_data(),
									total_neurons_,
									v_membranes_->mutable_gpu_data(),
									stream);
	}
}

template<typename T, typename T2>
unsigned int BrainBlock<T, T2>::update_F_input_spike_gpu(const unsigned int timestamp, 
															const unsigned int offset,
															hipStream_t stream)
{
	if(BLOCK_TYPE_INPUT == block_type_)
	{
		HIP_CHECK(hipMemsetAsync(f_inner_actives_->mutable_gpu_data(), 0x00, f_inner_actives_->size(), stream));
		if(offset < input_timestamps_->count())
		{
			//std::cout << "(" << timestamp << ", " << input_timestamps_->cpu_data()[offset] << ")" << std::endl;
			if(input_timestamps_->cpu_data()[offset] == timestamp)
			{
				const unsigned int n = input_rowptrs_->cpu_data()[offset + 1] - input_rowptrs_->cpu_data()[offset];
				if(n > 0)
				{
					update_input_spike_gpu(input_colinds_->gpu_data() + input_rowptrs_->cpu_data()[offset],
											n,
											f_inner_actives_->mutable_gpu_data(),
											stream);
				}
				return offset + 1;
			}
		}
	}
	return offset;
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_I_synaptic_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_synaptic_current_gpu<T, T2>(j_ex_presynaptics_->gpu_data(),
								j_in_presynaptics_->gpu_data(),
								g_ex_conducts_->gpu_data(),
								g_in_conducts_->gpu_data(),
								v_ex_membranes_->gpu_data(),
								v_in_membranes_->gpu_data(),
								v_membranes_->gpu_data(),
								total_neurons_,
								i_synaptics_->mutable_gpu_data(),
								stream);
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_V_membrane_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_membrane_voltage_gpu<T>(i_synaptics_->gpu_data(),
									i_ext_stimuli_->gpu_data(),
									v_resets_->gpu_data(),
									v_thresholds_->gpu_data(),
									c_membrane_reciprocals_->gpu_data(),
									v_leakages_->gpu_data(),
									g_leakages_->gpu_data(),
									t_refs_->gpu_data(),
									total_neurons_,
									delta_t_,
									t_,
									f_inner_actives_->mutable_gpu_data(),
									t_actives_->mutable_gpu_data(),
									v_membranes_->mutable_gpu_data(),
									stream);
	}
	else
	{
		update_membrane_voltage_for_input_gpu<T>(v_resets_->gpu_data(),
												v_thresholds_->gpu_data(),
												f_inner_actives_->gpu_data(),
												total_neurons_,
												v_membranes_->mutable_gpu_data(),
												stream);
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_F_active_gpu(const T a,
											   const T b,
											   bool saving_sample,
											   hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(saving_sample)
		{
			if(nullptr == uniform_samples_)
			{
				uniform_samples_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
			}
			
			update_spike_gpu<T>(gen_states_->mutable_gpu_data(),
								total_neurons_,
								noise_rates_->gpu_data(),
								f_inner_actives_->mutable_gpu_data(),
								a,
								b,
								uniform_samples_->mutable_gpu_data(),
								stream);
		}
		else
		{
			update_spike_gpu<T>(gen_states_->mutable_gpu_data(),
								total_neurons_,
								noise_rates_->gpu_data(),
								f_inner_actives_->mutable_gpu_data(),
								a,
								b,
								NULL,
								stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::record_F_sending_actives(const map<unsigned short, tuple<unsigned int*, int>>& rank_map)
{
	if(rank_map.empty())
		return;

	unsigned int n = 0;
	vector<unsigned int> sending_rowptrs;
	vector<unsigned int> sending_colinds;
	sending_rowptrs.push_back(n);
	
	for (auto it = rank_map.begin(); it != rank_map.end(); ++it)
	{
		f_sending_bids_.push_back(it->first);
		unsigned int count = static_cast<unsigned int>(std::get<1>(it->second));
		n += count;
		sending_rowptrs.push_back(n);
	}
	
	f_sending_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), sending_rowptrs.size() * sizeof(unsigned int));
	f_sending_colinds_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), n * sizeof(unsigned int));
	memcpy(f_sending_rowptrs_->mutable_cpu_data(), sending_rowptrs.data(), f_sending_rowptrs_->size());
	HIP_CHECK(hipMemcpy(f_sending_rowptrs_->mutable_gpu_data(), f_sending_rowptrs_->cpu_data(), f_sending_rowptrs_->size(), hipMemcpyHostToDevice));
	
	unsigned int blocks = divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD);
	f_sending_block_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), 2 * (blocks + 1) * sizeof(unsigned int));
	f_sending_block_rowptrs_->gpu_data();
	f_sending_active_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), 2 * sending_rowptrs.size() * sizeof(unsigned int));
	f_sending_active_rowptrs_->gpu_data();
	f_sending_active_rowptrs_->cpu_data();

	size_t storage_size_bytes = 0;
	count_F_sending_actives_temporary_storage_size(n,
												sending_rowptrs.size() - 1,
												f_sending_block_rowptrs_->mutable_gpu_data(),
												f_sending_active_rowptrs_->mutable_gpu_data(),
												storage_size_bytes);
	
	f_sending_active_colinds_= make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), storage_size_bytes);
	f_sending_active_colinds_->gpu_data();
	f_sending_active_colinds_->cpu_data();
	f_sending_actives_= make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), n * sizeof(unsigned char));
	f_sending_actives_->gpu_data();

	n = 0;
	for (auto it = rank_map.begin(); it != rank_map.end(); ++it, n++)
	{
		assert(static_cast<unsigned int>(std::get<1>(it->second)) == (sending_rowptrs[n + 1] - sending_rowptrs[n]));
		HIP_CHECK(hipMemcpy(f_sending_colinds_->mutable_gpu_data() + sending_rowptrs[n], std::get<0>(it->second), std::get<1>(it->second) * sizeof(unsigned int), hipMemcpyHostToDevice));
	}
	
	#if DEBUG
	HIP_CHECK(hipMemcpy(f_sending_colinds_->mutable_cpu_data(), f_sending_colinds_->gpu_data(), f_sending_colinds_->size(), hipMemcpyDeviceToHost));
	#endif
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_spikes_gpu(const unsigned int* sending_colinds,
													const unsigned int sending_num,
													const unsigned int f_active_num,
													unsigned char* f_actives,
													hipStream_t stream)
{
	HIP_CHECK(hipMemsetAsync(f_actives, 
							0x00,
							f_active_num * sizeof(unsigned char),
							stream));
	update_routing_spikes_gpu(sending_colinds, 
							sending_num, 
							f_actives,
							stream);
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_actives_gpu(const unsigned char* f_actives,
														const unsigned int* sending_rowptrs,
														const unsigned int* sending_colinds,
														const unsigned int segments,
														const unsigned int sending_count,
														unsigned char* f_sending_actives,
														unsigned int* block_rowptrs,
														unsigned int* active_rowptrs,
														unsigned int* active_colinds,
														hipStream_t stream)
{
	update_sending_spikes_gpu(f_actives,
							sending_rowptrs,
							sending_colinds,
							segments,
							sending_count,
							f_sending_actives,
							block_rowptrs,
							active_rowptrs,
							active_colinds,
							stream);
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::count_F_sending_actives_temporary_storage_size(const unsigned int sending_count,
																			const unsigned int segments,
																			unsigned int* block_rowptrs,
																			unsigned int* active_rowptrs,
																			size_t& storage_size_bytes,
																			hipStream_t stream)
{
	
	count_sending_spikes_temporary_storage_size(sending_count,
											segments,
											block_rowptrs,
											active_rowptrs,
											storage_size_bytes,
											stream);
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_sending_actives_gpu(hipStream_t stream)
{
	if(!f_sending_bids_.empty())
	{
		update_sending_spikes_gpu(f_inner_actives_->gpu_data(),
								f_sending_rowptrs_->gpu_data(),
								f_sending_colinds_->gpu_data(),
								f_sending_rowptrs_->count() - 1,
								f_sending_colinds_->count(),
								f_sending_actives_->mutable_gpu_data(),
								f_sending_block_rowptrs_->mutable_gpu_data(),
								f_sending_active_rowptrs_->mutable_gpu_data(),
								f_sending_active_colinds_->mutable_gpu_data(),
								stream);
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_offsets_gpu(const unsigned int* d_unions,
															const unsigned int union_num,
															const unsigned int offset_num,
															unsigned int* d_offsets,
															hipStream_t stream)
{
	update_routing_offsets_gpu(d_unions, 
								union_num,
								offset_num,
								d_offsets,
								stream);
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_neuron_ids_gpu(const unsigned int* d_unions,
															const unsigned int colind_num,
															unsigned int* d_colinds,
															hipStream_t stream)
{
	update_routing_neuron_ids_gpu(d_unions, 
								colind_num,
								d_colinds,
								stream);
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_recving_actives_gpu(hipStream_t stream)
{
	update_recving_spikes_gpu(f_receiving_rowptrs_->gpu_data(), 
							f_receiving_active_rowptrs_->gpu_data(),
							f_receiving_rowptrs_->count() - 1,
							f_receiving_active_colinds_->mutable_gpu_data(),
							stream);
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::update_J_presynaptic_gpu(bool saving_sample,
													hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_presynaptic_voltage_gpu<T, T2>(tao_ex_constants_->gpu_data(),
											tao_in_constants_->gpu_data(),
											total_neurons_,
											j_ex_presynaptics_->mutable_gpu_data(),
											j_ex_presynaptic_deltas_->mutable_gpu_data(),
											j_in_presynaptics_->mutable_gpu_data(),
											j_in_presynaptic_deltas_->mutable_gpu_data(),
											stream);
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_J_presynaptic_inner_gpu(hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		HIP_CHECK(hipMemsetAsync(j_ex_presynaptic_deltas_->mutable_gpu_data(), 0x00, j_ex_presynaptic_deltas_->size(), stream));
		HIP_CHECK(hipMemsetAsync(j_in_presynaptic_deltas_->mutable_gpu_data(), 0x00, j_in_presynaptic_deltas_->size(), stream));
		if(nullptr != inner_conninds_)
		{
			update_presynaptic_voltage_inner_gpu<T, T2>(inner_rowptrs_->gpu_data(),
													inner_colinds_->gpu_data(),
													inner_w_synaptics_->gpu_data(),
													inner_connkinds_->gpu_data(),
													inner_conninds_->count(),
													inner_conninds_->gpu_data(),
													f_inner_actives_->gpu_data(),
													j_ex_presynaptic_deltas_->mutable_gpu_data(),
													j_in_presynaptic_deltas_->mutable_gpu_data(),
													stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_J_presynaptic_outer_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(!f_receiving_bids_.empty())
		{
			assert(nullptr != f_receiving_active_rowptrs_ && f_receiving_active_rowptrs_->count() > 0);
			unsigned int n = f_receiving_active_rowptrs_->cpu_data()[f_receiving_active_rowptrs_->count() - 1];
			if(n > 0)
			{
				update_presynaptic_voltage_outer_gpu<T, T2>(outer_rowptrs_->gpu_data(),
													outer_colinds_->gpu_data(),
													outer_w_synaptics_->gpu_data(),
													outer_connkinds_->gpu_data(),
													f_receiving_active_colinds_->gpu_data(),
													n,
													j_ex_presynaptic_deltas_->mutable_gpu_data(),
													j_in_presynaptic_deltas_->mutable_gpu_data(),
													stream);
			}
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Vmeans_and_Imeans_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != sub_binfos_)
	{
		assert(imeans_->count() == sub_binfos_->count() &&
			vmeans_->count() == sub_binfos_->count());

		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_binfos_->gpu_data(),
									NULL,
									sub_binfos_->count(),
									NULL,
									v_membranes_->gpu_data(),
									i_synaptics_->gpu_data(),
									vmeans_->mutable_gpu_data(),
									imeans_->mutable_gpu_data(),
									stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_binfos_->gpu_data(),
									f_exclusive_counts_->gpu_data(),
									sub_binfos_->count(),
									f_exclusive_flags_->gpu_data(),
									v_membranes_->gpu_data(),
									i_synaptics_->gpu_data(),
									vmeans_->mutable_gpu_data(),
									imeans_->mutable_gpu_data(),
									stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Freqs_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != sub_binfos_)
	{
		assert(freqs_->count() == sub_binfos_->count());
		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_freqs_gpu(sub_binfos_->gpu_data(),
							sub_binfos_->count(),
							NULL,
							f_inner_actives_->gpu_data(),
							freqs_->mutable_gpu_data(),
							stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_freqs_gpu(sub_binfos_->gpu_data(),
							sub_binfos_->count(),
							f_exclusive_flags_->gpu_data(),
							f_inner_actives_->gpu_data(),
							freqs_->mutable_gpu_data(),
							stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Vmeans_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != sub_binfos_)
	{
		assert(vmeans_->count() == sub_binfos_->count());
		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_binfos_->gpu_data(),
										NULL,
										sub_binfos_->count(),
										NULL,
										v_membranes_->gpu_data(),
										NULL,
										vmeans_->mutable_gpu_data(),
										NULL,
										stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_binfos_->gpu_data(),
										f_exclusive_counts_->gpu_data(),
										sub_binfos_->count(),
										f_exclusive_flags_->gpu_data(),
										v_membranes_->gpu_data(),
										NULL,
										vmeans_->mutable_gpu_data(),
										NULL,
										stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Imeans_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != sub_binfos_)
	{
		assert(imeans_->count() == sub_binfos_->count());
		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_binfos_->gpu_data(),
										NULL,
										sub_binfos_->count(),
										NULL,
										NULL,
										i_synaptics_->gpu_data(),
										NULL,
										imeans_->mutable_gpu_data(),
										stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_binfos_->gpu_data(),
										f_exclusive_counts_->gpu_data(),
										sub_binfos_->count(),
										f_exclusive_flags_->gpu_data(),
										NULL,
										i_synaptics_->gpu_data(),
										NULL,
										imeans_->mutable_gpu_data(),
										stream);
		}
	}
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Spikes_and_Vmembs_gpu(const unsigned int* samples,
															const unsigned int n,
															char* spikes,
															T* vmembs,
															hipStream_t stream)
{
	stat_spikes_and_vmembs_gpu<T>(samples,
								  n,
								  f_inner_actives_->gpu_data(),
								  v_membranes_->gpu_data(),
								  spikes,
								  vmembs,
								  stream);
}

template<typename T, typename T2>
void BrainBlock<T, T2>::transform_J_presynaptic_gpu(T* j_presynaptics,
														hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	transform_presynaptic_voltage_gpu(j_ex_presynaptics_->gpu_data(),
									j_in_presynaptics_->gpu_data(),
									total_neurons_,
									j_presynaptics,
									stream);
}

template<typename T, typename T2>
void BrainBlock<T, T2>::add_sample_neurons_gpu(unsigned int* samples, const unsigned int n)
{
	assert(n > 0 && n < total_neurons_);
	for(unsigned int i = 0; i < n; i++)
		assert(samples[i] < total_neurons_);

	sample_indices_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), n * sizeof(unsigned int));
	f_sample_actives_ = make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), n * sizeof(unsigned char));
	v_sample_membranes_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), n * sizeof(T));

	HIP_CHECK(hipMemcpy(sample_indices_->mutable_gpu_data(), samples, sample_indices_->size(), hipMemcpyHostToDevice));
	f_sample_actives_->gpu_data();
	v_sample_membranes_->gpu_data();
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_sample_neurons_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != sample_indices_)
	{
		update_sample_neuron_gpu<T>(f_inner_actives_->gpu_data(),
								v_membranes_->gpu_data(),
								sample_indices_->gpu_data(),
								sample_indices_->count(),
								f_sample_actives_->mutable_gpu_data(),
								v_sample_membranes_->mutable_gpu_data(),
								stream);
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_Props_gpu(const unsigned int* neruon_indice,
											const unsigned int* prop_indice,
											const float* prop_vals,
											const unsigned int n,
											hipStream_t stream)
{
	Properties<T, T2> prop;
	prop.n = total_neurons_;
	prop.i_ext_stimuli = i_ext_stimuli_->mutable_gpu_data();
	prop.c_membrane_reciprocals = c_membrane_reciprocals_->mutable_gpu_data();
	prop.t_refs = t_refs_->mutable_gpu_data();
	prop.g_leakages = g_leakages_->mutable_gpu_data();
	prop.v_leakages = v_leakages_->mutable_gpu_data();
	prop.v_thresholds = v_thresholds_->mutable_gpu_data();
	prop.v_resets = v_resets_->mutable_gpu_data();
	prop.g_ex_conducts = g_ex_conducts_->mutable_gpu_data();
	prop.g_in_conducts = g_in_conducts_->mutable_gpu_data();
	prop.v_ex_membranes = v_ex_membranes_->mutable_gpu_data();
	prop.v_in_membranes = v_in_membranes_->mutable_gpu_data();
	prop.tao_ex_constants = tao_ex_constants_->mutable_gpu_data();
	prop.tao_in_constants = tao_in_constants_->mutable_gpu_data();
	prop.noise_rates = noise_rates_->mutable_gpu_data();
	update_props_gpu<T, T2>(neruon_indice, prop_indice, prop_vals, n, prop, stream);
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_Prop_Cols_gpu(const unsigned int* prop_indice,
											const unsigned int* brain_indice,
											const T* hp_vals,
											const unsigned int n,
											hipStream_t stream)
{
	Properties<T, T2> prop;
	prop.n = total_neurons_;
	prop.i_ext_stimuli = i_ext_stimuli_->mutable_gpu_data();
	prop.c_membrane_reciprocals = c_membrane_reciprocals_->mutable_gpu_data();
	prop.t_refs = t_refs_->mutable_gpu_data();
	prop.g_leakages = g_leakages_->mutable_gpu_data();
	prop.v_leakages = v_leakages_->mutable_gpu_data();
	prop.v_thresholds = v_thresholds_->mutable_gpu_data();
	prop.v_resets = v_resets_->mutable_gpu_data();
	prop.g_ex_conducts = g_ex_conducts_->mutable_gpu_data();
	prop.g_in_conducts = g_in_conducts_->mutable_gpu_data();
	prop.v_ex_membranes = v_ex_membranes_->mutable_gpu_data();
	prop.v_in_membranes = v_in_membranes_->mutable_gpu_data();
	prop.tao_ex_constants = tao_ex_constants_->mutable_gpu_data();
	prop.tao_in_constants = tao_in_constants_->mutable_gpu_data();
	prop.noise_rates = noise_rates_->mutable_gpu_data();
	update_prop_cols_gpu<T, T2>(sub_bids_->gpu_data(),
							sub_binfos_->gpu_data(),
							sub_binfos_->count(),
							prop_indice,
							brain_indice,
							hp_vals,
							n,
							prop,
							stream);
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_Gamma_Prop_Cols_gpu(const unsigned int* prop_indice,
															const unsigned int* brain_indice,
															const T* alphas,
															const T* betas,
															const unsigned int n,
															hipStream_t stream)
{
	assert(n > 0);
	unsigned long long offset = philox_seed_offset_.fetch_add(10);
	Properties<T, T2> prop;

	prop.n = total_neurons_;
	prop.i_ext_stimuli = i_ext_stimuli_->mutable_gpu_data();
	prop.c_membrane_reciprocals = c_membrane_reciprocals_->mutable_gpu_data();
	prop.t_refs = t_refs_->mutable_gpu_data();
	prop.g_leakages = g_leakages_->mutable_gpu_data();
	prop.v_leakages = v_leakages_->mutable_gpu_data();
	prop.v_thresholds = v_thresholds_->mutable_gpu_data();
	prop.v_resets = v_resets_->mutable_gpu_data();
	prop.g_ex_conducts = g_ex_conducts_->mutable_gpu_data();
	prop.g_in_conducts = g_in_conducts_->mutable_gpu_data();
	prop.v_ex_membranes = v_ex_membranes_->mutable_gpu_data();
	prop.v_in_membranes = v_in_membranes_->mutable_gpu_data();
	prop.tao_ex_constants = tao_ex_constants_->mutable_gpu_data();
	prop.tao_in_constants = tao_in_constants_->mutable_gpu_data();
	prop.noise_rates = noise_rates_->mutable_gpu_data();

	gamma_gpu<T, T2>(sub_bids_->gpu_data(),
				sub_binfos_->gpu_data(),
				sub_binfos_->count(),
				prop_indice,
				brain_indice,
				alphas,
				betas,
				n,
				seed_,
				offset,
				prop,
			 	stream);
	
}

template<typename T, typename T2>
void BrainBlock<T, T2>::init_all_params_and_stages_cpu()
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		HIP_CHECK(hipMemcpy(j_ex_presynaptics_->mutable_cpu_data(), j_ex_presynaptics_->gpu_data(), j_ex_presynaptics_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(j_in_presynaptics_->mutable_cpu_data(), j_in_presynaptics_->gpu_data(), j_in_presynaptics_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(t_actives_->mutable_cpu_data(), t_actives_->gpu_data(), t_actives_->size(), hipMemcpyDeviceToHost));

		if(nullptr != inner_conninds_)
		{
			HIP_CHECK(hipMemcpy(inner_rowptrs_->mutable_cpu_data(), inner_rowptrs_->gpu_data(), inner_rowptrs_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(inner_colinds_->mutable_cpu_data(), inner_colinds_->gpu_data(), inner_colinds_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(inner_w_synaptics_->mutable_cpu_data(), inner_w_synaptics_->gpu_data(), inner_w_synaptics_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(inner_connkinds_->mutable_cpu_data(), inner_connkinds_->gpu_data(), inner_connkinds_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(inner_conninds_->mutable_cpu_data(), inner_conninds_->gpu_data(), inner_conninds_->size(), hipMemcpyDeviceToHost));
		}
		
		if(nullptr != outer_rowptrs_)
		{
			HIP_CHECK(hipMemcpy(outer_rowptrs_->mutable_cpu_data(), outer_rowptrs_->gpu_data(), outer_rowptrs_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(outer_colinds_->mutable_cpu_data(), outer_colinds_->gpu_data(), outer_colinds_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(outer_w_synaptics_->mutable_cpu_data(), outer_w_synaptics_->gpu_data(), outer_w_synaptics_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(outer_connkinds_->mutable_cpu_data(), outer_connkinds_->gpu_data(), outer_connkinds_->size(), hipMemcpyDeviceToHost));
		}

		HIP_CHECK(hipMemcpy(noise_rates_->mutable_cpu_data(), noise_rates_->gpu_data(), noise_rates_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(i_ext_stimuli_->mutable_cpu_data(), i_ext_stimuli_->gpu_data(), i_ext_stimuli_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(c_membrane_reciprocals_->mutable_cpu_data(), c_membrane_reciprocals_->gpu_data(), c_membrane_reciprocals_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(t_refs_->mutable_cpu_data(), t_refs_->gpu_data(), t_refs_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(g_leakages_->mutable_cpu_data(), g_leakages_->gpu_data(), g_leakages_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(v_leakages_->mutable_cpu_data(), v_leakages_->gpu_data(), v_leakages_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(v_thresholds_->mutable_cpu_data(), v_thresholds_->gpu_data(), v_thresholds_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(v_resets_->mutable_cpu_data(), v_resets_->gpu_data(), v_resets_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(g_ex_conducts_->mutable_cpu_data(), g_ex_conducts_->gpu_data(), g_ex_conducts_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(g_in_conducts_->mutable_cpu_data(), g_in_conducts_->gpu_data(), g_in_conducts_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(v_ex_membranes_->mutable_cpu_data(), v_ex_membranes_->gpu_data(), v_ex_membranes_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(v_in_membranes_->mutable_cpu_data(), v_in_membranes_->gpu_data(), v_in_membranes_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(tao_ex_constants_->mutable_cpu_data(), tao_ex_constants_->gpu_data(), tao_ex_constants_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(tao_in_constants_->mutable_cpu_data(), tao_in_constants_->gpu_data(), tao_in_constants_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(v_membranes_->mutable_cpu_data(), v_membranes_->gpu_data(), v_membranes_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(f_inner_actives_->mutable_cpu_data(), f_inner_actives_->gpu_data(), f_inner_actives_->size(), hipMemcpyDeviceToHost));

	}

	if(nullptr != sub_binfos_)
	{
		HIP_CHECK(hipMemcpy(sub_binfos_->mutable_cpu_data(), sub_binfos_->gpu_data(), sub_binfos_->size(), hipMemcpyDeviceToHost));
		freqs_->cpu_data();
		vmeans_->cpu_data();
		imeans_->cpu_data();
	}

	if(nullptr != f_exclusive_flags_)
	{
		assert(nullptr != f_exclusive_counts_);
		HIP_CHECK(hipMemcpy(f_exclusive_flags_->mutable_cpu_data(), f_exclusive_flags_->gpu_data(), f_exclusive_flags_->size(), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(f_exclusive_counts_->mutable_cpu_data(), f_exclusive_counts_->gpu_data(), f_exclusive_counts_->size(), hipMemcpyDeviceToHost));
	}
	if(nullptr != sample_indices_)
	{
		v_sample_membranes_->cpu_data();
	}
	
	if(nullptr != sample_indices_)
	{
		HIP_CHECK(hipMemcpy(sample_indices_->mutable_cpu_data(), sample_indices_->gpu_data(), sample_indices_->size(), hipMemcpyDeviceToHost));
		f_sample_actives_->cpu_data();
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::reset_V_membrane_cpu()
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		assert(v_membranes_->size() == v_resets_->size());
		memcpy(v_membranes_->mutable_cpu_data(), v_resets_->cpu_data(), v_membranes_->size());
	}
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_I_synaptic_cpu()
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		for(unsigned int i = 0; i < total_neurons_; i++)
		{
			T vi = (v_membranes_->cpu_data())[i];
		
			T2 ji_ex = (j_ex_presynaptics_->cpu_data())[i];
			T2 gi_ex = (g_ex_conducts_->cpu_data())[i];
			T2 vi_ex = (v_ex_membranes_->cpu_data())[i];

			T2 ji_in = (j_in_presynaptics_->cpu_data())[i];
			T2 gi_in = (g_in_conducts_->cpu_data())[i];
			T2 vi_in = (v_in_membranes_->cpu_data())[i];

			T sum = gi_ex.x * (vi_ex.x - vi) * ji_ex.x;
			sum += gi_ex.y * (vi_ex.y - vi) * ji_ex.y;
			sum += gi_in.x * (vi_in.x - vi) * ji_in.x;
			sum += gi_in.y * (vi_in.y - vi) * ji_in.y;
			(i_synaptics_->mutable_cpu_data())[i] = sum;
		}
	}
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_V_membrane_cpu()
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		for(unsigned int i = 0; i < total_neurons_; i++)
		{
			unsigned char f_changed = 0x00;
			unsigned char v_changed = 0x00;
			
			T vi;
			unsigned char fi = (f_inner_actives_->cpu_data())[i];
			if(fi)
			{
				f_changed = 0x01;
				v_changed = 0x01;
				vi = (v_resets_->cpu_data())[i];
				fi = 0x00;
			}
			
			if(t_ >= ((t_actives_->cpu_data())[i] + (t_refs_->cpu_data())[i]))
			{
				vi = (v_membranes_->cpu_data())[i];
				T cvj = (g_leakages_->cpu_data())[i] * ((v_leakages_->cpu_data())[i] - vi) + 
						(i_synaptics_->cpu_data())[i] + (i_ext_stimuli_->cpu_data())[i];
				vi += delta_t_ * (c_membrane_reciprocals_->cpu_data())[i] * cvj;
				
				T vth = (v_thresholds_->cpu_data())[i];
				if(vi >= vth)
				{
					vi = vth;
					fi = 0x01;
					f_changed = 0x01;
					(t_actives_->mutable_cpu_data())[i] = t_;
				}
				
				v_changed = 0x01;
			}
			
			if(f_changed)
				(f_inner_actives_->mutable_cpu_data())[i] = fi;

			if(v_changed)
				(v_membranes_->mutable_cpu_data())[i] = vi;
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_F_active_cpu()
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		
		HIP_CHECK(hipMemcpy(uniform_samples_->mutable_cpu_data(), uniform_samples_->gpu_data(), uniform_samples_->size(), hipMemcpyDeviceToHost));
		unsigned char* flags = f_inner_actives_->mutable_cpu_data();
		for(unsigned int i = 0; i < total_neurons_; i++)
		{
			unsigned char flag = flags[i];
			flag |= static_cast<unsigned char>((uniform_samples_->cpu_data())[i] < noise_rates_->cpu_data()[i]);
			flags[i] = flag;
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_sending_actives_cpu()
{
	if(!f_sending_bids_.empty())
	{
		f_sending_active_rowptrs_->mutable_cpu_data()[0] = 0;
		for(unsigned int i = 0; i < f_sending_bids_.size(); i++)
		{
			unsigned int k = 0;
			for(unsigned int idx = 0, j = f_sending_rowptrs_->cpu_data()[i]; j < f_sending_rowptrs_->cpu_data()[i + 1]; j++, idx++)
			{
				unsigned int nidx = f_sending_colinds_->cpu_data()[j];
				unsigned char flag = f_inner_actives_->cpu_data()[nidx];
				if(flag)
				{
					f_sending_active_colinds_->mutable_cpu_data()[k + f_sending_active_rowptrs_->cpu_data()[i]] = idx;
					k++;
				}
			}
			f_sending_active_rowptrs_->mutable_cpu_data()[i + 1] = k + f_sending_active_rowptrs_->cpu_data()[i];
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_J_presynaptic_cpu()
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		for(unsigned int i = 0; i < total_neurons_; i++)
		{
			T2 ji_ex = (j_ex_presynaptics_->cpu_data())[i];
			T2 ji_ex_delta = (j_ex_presynaptic_deltas_->cpu_data())[i];
			T2 tao_ex = (tao_ex_constants_->cpu_data())[i];
			
			ji_ex.x = ji_ex.x * tao_ex.x + ji_ex_delta.x;
			ji_ex.y = ji_ex.y * tao_ex.y + ji_ex_delta.y;

			T2 ji_in = (j_in_presynaptics_->cpu_data())[i];
			T2 ji_in_delta = (j_in_presynaptic_deltas_->cpu_data())[i];
			T2 tao_in = (tao_in_constants_->cpu_data())[i];
			
			ji_in.x = ji_in.x * tao_in.x + ji_in_delta.x;
			ji_in.y = ji_in.y * tao_in.y + ji_in_delta.y;
			
			(j_ex_presynaptics_->mutable_cpu_data())[i] = ji_ex;
			(j_in_presynaptics_->mutable_cpu_data())[i] = ji_in;
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_J_presynaptic_inner_cpu()
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		memset(j_ex_presynaptic_deltas_->mutable_cpu_data(), 0x00, j_ex_presynaptic_deltas_->size());
		memset(j_in_presynaptic_deltas_->mutable_cpu_data(), 0x00, j_in_presynaptic_deltas_->size());
		if(nullptr != inner_conninds_)
		{
			for(unsigned int i = 0; i < inner_conninds_->count(); i++)
			{
				unsigned int fidx = (inner_conninds_->cpu_data())[i];
				unsigned char flag = (f_inner_actives_->cpu_data())[fidx];
				if(flag)
				{
					unsigned int begin = (inner_rowptrs_->cpu_data())[i];
					unsigned int end = (inner_rowptrs_->cpu_data())[i + 1];
					for(unsigned int j = begin; j < end; j++)
					{
						unsigned int nidx = (inner_colinds_->cpu_data())[j];
						T2 weight = (inner_w_synaptics_->cpu_data())[j];
						T2* val;
						flag = (inner_connkinds_->cpu_data())[j];
						
						if(flag)
							val = j_in_presynaptic_deltas_->mutable_cpu_data();
						else
							val = j_ex_presynaptic_deltas_->mutable_cpu_data();
						
						val[nidx].x += weight.x;
						val[nidx].y += weight.y;
					}
				}
			}
		}
	}
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_J_presynaptic_outer_cpu()
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(!f_receiving_bids_.empty())
		{
			unsigned int count = f_receiving_active_rowptrs_->cpu_data()[f_receiving_active_rowptrs_->count() - 1];
			for(unsigned int i = 0; i < count; i++)
			{
				unsigned int fidx = f_receiving_active_colinds_->cpu_data()[i];
				
				unsigned int begin = (outer_rowptrs_->cpu_data())[fidx];
				unsigned int end = (outer_rowptrs_->cpu_data())[fidx + 1];
				for(unsigned int j = begin; j < end; j++)
				{
					unsigned int nidx = (outer_colinds_->cpu_data())[j];
					T2 weight = (outer_w_synaptics_->cpu_data())[j];
					T2* val;
					unsigned char flag = (outer_connkinds_->cpu_data())[j];
					
					if(flag)
						val = j_in_presynaptic_deltas_->mutable_cpu_data();
					else
						val = j_ex_presynaptic_deltas_->mutable_cpu_data();
					
					val[nidx].x += weight.x;
					val[nidx].y += weight.y;
				}
			}
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::stat_Freqs_cpu()
{
	if(nullptr != sub_binfos_)
	{	
		for(unsigned int i = 0; i < sub_binfos_->count(); i++)
		{
			unsigned char exclusive_flag = 0x00;
			
			uint2& info = (sub_binfos_->mutable_cpu_data())[i];
			unsigned int fsum = 0;
			unsigned int end = info.x + info.y;
			for(unsigned int j = info.x; j < end; j++)
			{
				if(nullptr != f_exclusive_flags_)
				{
					exclusive_flag = f_exclusive_flags_->cpu_data()[j];
				}
				
				fsum += (f_inner_actives_without_noise_->cpu_data())[j] * ((exclusive_flag == 0x00) ? (T)1 : (T)0);
			}

			(freqs_->mutable_cpu_data())[i] = fsum;
		}
	}
}


template<typename T, typename T2>
void BrainBlock<T, T2>::stat_Vmeans_and_Imeans_cpu(bool has_vmean,
														bool has_imean)
{
	if(!has_vmean && !has_imean)
		return;
	
	if(nullptr != sub_binfos_)
	{	
		for(unsigned int i = 0; i < sub_binfos_->count(); i++)
		{
			unsigned char exclusive_flag = 0x00;
			unsigned int exclusive_count = 0;
			if(nullptr != f_exclusive_counts_)
			{
				exclusive_count = f_exclusive_counts_->cpu_data()[i];
			}
			
			uint2& info = (sub_binfos_->mutable_cpu_data())[i];
			T vsum = static_cast<T>(0);
			T isum = static_cast<T>(0);
			unsigned int end = info.x + info.y;
			for(unsigned int j = info.x; j < end; j++)
			{
				if(nullptr != f_exclusive_flags_)
				{
					exclusive_flag = f_exclusive_flags_->cpu_data()[j];
				}

				if(has_vmean)
				{
					vsum += (v_membranes_->cpu_data())[j] * ((exclusive_flag == 0x00) ? (T)1 : (T)0);
				}

				if(has_imean)
				{
					isum += (i_synaptics_->cpu_data())[j] * ((exclusive_flag == 0x00) ? (T)1 : (T)0);
				}
			}
			
			if(info.y > 0)
			{
				assert(info.y > exclusive_count);
				if(has_vmean)
				{
					vsum /= (info.y - exclusive_count);
				}

				if(has_imean)
				{
					isum /= (info.y - exclusive_count);
				}
			}
			else
			{
				assert(0 == exclusive_count);
			}

			if(has_vmean)
			{
				(vmeans_->mutable_cpu_data())[i] = vsum;
			}

			if(has_imean)
			{
				(imeans_->mutable_cpu_data())[i] = isum;
			}
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::stat_Spikes_and_Vmembs_cpu(const unsigned int* samples,
														const unsigned int n,
														char* spikes,
														T* vmembs)
{
	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int sidx = samples[i];
		assert(sidx < total_neurons_);
		if(NULL != spikes)
		{
			spikes[i] = f_inner_actives_->cpu_data()[sidx];
		}

		if(NULL != vmembs)
		{
			vmembs[i] = v_membranes_->cpu_data()[sidx];
		}
		
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_spikes_cpu(const unsigned int* sending_colinds,
														const unsigned int sending_counts,
														const unsigned int n,
														unsigned char* f_actives)
{
	memset(f_actives, 
			0x00,
			n * sizeof(unsigned char));
	
	for(unsigned int i = 0; i < sending_counts; i++)
	{
		unsigned int nidx = sending_colinds[i];
		f_actives[nidx] = 0x01;
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_actives_cpu(const unsigned char* f_actives,
														const unsigned int* sending_rowptrs,
														const unsigned int* sending_colinds,
														const unsigned int segments,
														unsigned int* f_active_rowptrs,
														unsigned int* f_active_colinds)
{
	f_active_rowptrs[0] = 0;
	assert(sending_rowptrs[0] == 0);
	for(unsigned int i = 0; i < segments; i++)
	{
		unsigned int k = 0;
		for(unsigned int idx = 0, j = sending_rowptrs[i]; j < sending_rowptrs[i + 1]; idx++, j++)
		{
			unsigned int nidx = sending_colinds[j];
			unsigned char flag = f_actives[nidx];
			if(flag)
			{
				f_active_colinds[f_active_rowptrs[i] + k] = idx;
				k++;
			}
		}
		f_active_rowptrs[i + 1] = f_active_rowptrs[i] + k;
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_offsets_cpu(const unsigned int* unions,
															const unsigned int union_num,
															const unsigned int rows,
															const unsigned int* rowptrs,
															unsigned int* colinds)
{
	unsigned int j = 0;
	for(unsigned int i = 0; i < rows; i++)
	{
		assert(j == rowptrs[i]);
		for(; j < rowptrs[i + 1]; j++)
		{
			const unsigned int* iter = thrust::lower_bound(unions, unions + union_num, colinds[j]);
			unsigned int idx = iter - unions;
			assert(idx < union_num);
			colinds[j] = idx;
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_routing_neuron_ids_cpu(const unsigned int* unions,
															const unsigned int colind_num,
															unsigned int* colinds)
{
	for(unsigned int i = 0; i < colind_num; i++)
	{
		unsigned int idx = colinds[i];
		colinds[i] = unions[idx];
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_recving_actives_cpu()
{
	if(!f_receiving_bids_.empty())
	{
		for(unsigned int i = 1; i < f_receiving_bids_.size(); i++)
		{
			unsigned int offset = f_receiving_rowptrs_->cpu_data()[i];
			for(unsigned int j = f_receiving_active_rowptrs_->cpu_data()[i]; j < f_receiving_active_rowptrs_->cpu_data()[i + 1]; j++)
			{
				f_receiving_active_colinds_->mutable_cpu_data()[j] += offset;
			}
		}
	}
}

/*
template<typename T, typename T2>
void BrainBlock<T, T2>::stat_MaxErr_V_membrane_for_spiking_neurons_cpu()
{
}

template<typename T, typename T2>
void BrainBlock<T, T2>::stat_Mean_I_synaptic_cpu()
{
}

template<typename T, typename T2>
void BrainBlock<T, T2>::stat_Variance_I_synaptic_cpu()
{
}
*/
template<typename T, typename T2>
void BrainBlock<T, T2>::update_sample_neurons_cpu()
{
	if(nullptr != sample_indices_)
	{
		for(unsigned int i = 0; i < sample_indices_->count(); i++)
		{
			unsigned int idx = (sample_indices_->cpu_data())[i];
			unsigned char flag = (f_inner_actives_->cpu_data())[idx];
			(f_sample_actives_->mutable_cpu_data())[i] = flag;
			(v_sample_membranes_->mutable_cpu_data())[i] = (v_membranes_->cpu_data())[idx];
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::fetch_props(const unsigned int* neuron_indice,
									const unsigned int* prop_indice,
									const unsigned int n,
									vector<T>& result)
{
	#define FETCH_T_TYPE(CASE, DATA, PROP) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		result[i] = DATA[nid]; \
	break;

	#define FETCH_T2_TYPE(CASE, DATA, PROP, FILED) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		result[i] = DATA[nid].FILED; \
	break;

	result.resize(n);
	vector<T> i_ext_stimuli;
	vector<T> c_membrane_reciprocals;
	vector<T> t_refs;
	vector<T> g_leakages;
	vector<T> v_leakages;
	vector<T> v_thresholds;
	vector<T> v_resets;
	vector<T2> g_ex_conducts;
	vector<T2> g_in_conducts;
	vector<T2> v_ex_membranes;
	vector<T2> v_in_membranes;
	vector<T2> tao_ex_constants;
	vector<T2> tao_in_constants;
	vector<T> noise_rates;
	
	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int nid = neuron_indice[i];
		unsigned int pid = prop_indice[i];
		switch(pid)
		{
			FETCH_T_TYPE(EXT_STIMULI_I, i_ext_stimuli, i_ext_stimuli_)
			FETCH_T_TYPE(MEMBRANE_C, c_membrane_reciprocals, c_membrane_reciprocals_)
			FETCH_T_TYPE(REF_T, t_refs, t_refs_)
			FETCH_T_TYPE(LEAKAGE_G, g_leakages, g_leakages_)
			FETCH_T_TYPE(LEAKAGE_V, v_leakages, v_leakages_)
			FETCH_T_TYPE(THRESHOLD_V, v_thresholds, v_thresholds_)
			FETCH_T_TYPE(RESET_V, v_resets, v_resets_)
			FETCH_T2_TYPE(CONDUCT_G_AMPA, g_ex_conducts, g_ex_conducts_, x)
			FETCH_T2_TYPE(CONDUCT_G_NMDA, g_ex_conducts, g_ex_conducts_, y)
			FETCH_T2_TYPE(CONDUCT_G_GABAa, g_in_conducts, g_in_conducts_, x)
			FETCH_T2_TYPE(CONDUCT_G_GABAb, g_in_conducts, g_in_conducts_, y)
			
			FETCH_T2_TYPE(MEMBRANE_V_AMPA, v_ex_membranes, v_ex_membranes_, x)
			FETCH_T2_TYPE(MEMBRANE_V_NMDA, v_ex_membranes, v_ex_membranes_, y)
			FETCH_T2_TYPE(MEMBRANE_V_GABAa, v_in_membranes, v_in_membranes_, x)
			FETCH_T2_TYPE(MEMBRANE_V_GABAb, v_in_membranes, v_in_membranes_, y)

			FETCH_T2_TYPE(TAO_AMPA, tao_ex_constants, tao_ex_constants_, x)
			FETCH_T2_TYPE(TAO_NMDA, tao_ex_constants, tao_ex_constants_, y)
			FETCH_T2_TYPE(TAO_GABAa, tao_in_constants, tao_in_constants_, x)
			FETCH_T2_TYPE(TAO_GABAb, tao_in_constants, tao_in_constants_, y)
			FETCH_T_TYPE(NOISE_RATE, noise_rates, noise_rates_)
			default:
				assert(0);
			break;
		}
	}

	#undef FETCH_T_TYPE
	#undef FETCH_T2_TYPE
}

template<typename T, typename T2>
void BrainBlock<T, T2>::fetch_prop_cols(const unsigned int* prop_indice,
										const unsigned int* brain_indice,
										const unsigned int n,
										vector<vector<T>>& results)
{
	#define FETCH_T_TYPE(CASE, DATA, PROP) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		results[i][total] = DATA[j]; \
	break;

	#define FETCH_T2_TYPE(CASE, DATA, PROP, FILED) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		results[i][total] = DATA[j].FILED; \
	break;

	std::vector<unsigned int> sub_bids(sub_bids_->count());
	std::vector<uint2> sub_binfos(sub_binfos_->count());
	HIP_CHECK(hipMemcpy(sub_bids.data(), sub_bids_->gpu_data(), sub_bids_->size(), hipMemcpyDeviceToHost));
	HIP_CHECK(hipMemcpy(sub_binfos.data(), sub_binfos_->gpu_data(), sub_binfos_->size(), hipMemcpyDeviceToHost));
	
	vector<T> i_ext_stimuli;
	vector<T> c_membrane_reciprocals;
	vector<T> t_refs;
	vector<T> g_leakages;
	vector<T> v_leakages;
	vector<T> v_thresholds;
	vector<T> v_resets;
	vector<T2> g_ex_conducts;
	vector<T2> g_in_conducts;
	vector<T2> v_ex_membranes;
	vector<T2> v_in_membranes;
	vector<T2> tao_ex_constants;
	vector<T2> tao_in_constants;
	vector<T> noise_rates;
	
	results.resize(n);
	for(unsigned int i = 0; i < n; i++)
	{
		const unsigned int pid = prop_indice[i];
		const unsigned int* iter = thrust::find(sub_bids.data(), sub_bids.data() + sub_bids.size(), brain_indice[i]);
		unsigned int bid = iter - sub_bids.data();
		if(bid == sub_bids.size())
			continue;
		
		uint2 info = sub_binfos[bid];
		results[i].resize(info.y);
		
		unsigned int total = 0;
		for(unsigned int j = info.x; j < (info.x + info.y); j++)
		{
			switch(pid)
			{
				FETCH_T_TYPE(EXT_STIMULI_I, i_ext_stimuli, i_ext_stimuli_)
				FETCH_T_TYPE(MEMBRANE_C, c_membrane_reciprocals, c_membrane_reciprocals_)
				FETCH_T_TYPE(REF_T, t_refs, t_refs_)
				FETCH_T_TYPE(LEAKAGE_G, g_leakages, g_leakages_)
				FETCH_T_TYPE(LEAKAGE_V, v_leakages, v_leakages_)
				FETCH_T_TYPE(THRESHOLD_V, v_thresholds, v_thresholds_)
				FETCH_T_TYPE(RESET_V, v_resets, v_resets_)
				FETCH_T2_TYPE(CONDUCT_G_AMPA, g_ex_conducts, g_ex_conducts_, x)
				FETCH_T2_TYPE(CONDUCT_G_NMDA, g_ex_conducts, g_ex_conducts_, y)
				FETCH_T2_TYPE(CONDUCT_G_GABAa, g_in_conducts, g_in_conducts_, x)
				FETCH_T2_TYPE(CONDUCT_G_GABAb, g_in_conducts, g_in_conducts_, y)
				
				FETCH_T2_TYPE(MEMBRANE_V_AMPA, v_ex_membranes, v_ex_membranes_, x)
				FETCH_T2_TYPE(MEMBRANE_V_NMDA, v_ex_membranes, v_ex_membranes_, y)
				FETCH_T2_TYPE(MEMBRANE_V_GABAa, v_in_membranes, v_in_membranes_, x)
				FETCH_T2_TYPE(MEMBRANE_V_GABAb, v_in_membranes, v_in_membranes_, y)

				FETCH_T2_TYPE(TAO_AMPA, tao_ex_constants, tao_ex_constants_, x)
				FETCH_T2_TYPE(TAO_NMDA, tao_ex_constants, tao_ex_constants_, y)
				FETCH_T2_TYPE(TAO_GABAa, tao_in_constants, tao_in_constants_, x)
				FETCH_T2_TYPE(TAO_GABAb, tao_in_constants, tao_in_constants_, y)
				FETCH_T_TYPE(NOISE_RATE, noise_rates, noise_rates_)
				
				default:
					assert(0);
				break;
			}  
			total++;
		}
	}

	#undef FETCH_T_TYPE
	#undef FETCH_T2_TYPE
}

template class BrainBlock<float, float2>;
}//namespace dtb

