#pragma once

#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <tuple>
#include <hiprand_kernel.h>
#include "common.hpp"
#include "data_allocator.hpp"

using namespace std;

//digital twin brain
namespace dtb {

template<typename T, typename T2>
class BrainBlock
{
public:

	enum BLOCK_TYPE
	{
		BLOCK_TYPE_INPUT = 0,
		BLOCK_TYPE_NORMAL
	};

	BrainBlock(const unsigned short block_id,
			const int gpu_id,
			const T delta_t = 0.1f,
			const T t = 0.f,
			const unsigned long long seed = gen_seed());
	
	~BrainBlock(){};

	void init_config_params_gpu(const std::string& filename);
	void init_connection_table_gpu(const std::string& filename);
	void init_all_stages_gpu();

	unsigned int update_F_input_spike_gpu(const unsigned int timestamp, 
										const unsigned int offset = 0,
										hipStream_t stream = NULL);
	void update_I_synaptic_gpu(hipStream_t stream = NULL);
	void update_V_membrane_gpu(hipStream_t stream = NULL);
	void update_F_active_gpu(const T a = 0.f,
								const T b = 1.f,
								bool saving_sample = false,
								hipStream_t stream = NULL);
	void update_J_presynaptic_inner_gpu(hipStream_t stream = NULL);
	void update_J_presynaptic_outer_gpu(hipStream_t stream = NULL);
	void update_J_presynaptic_gpu(bool saving_sample = false,
										hipStream_t stream = NULL);
	
	void reset_V_membrane_gpu(hipStream_t stream = NULL);
	void stat_Vmeans_and_Imeans_gpu(hipStream_t stream = NULL);

	void stat_Freqs_gpu(hipStream_t stream = NULL);

	void stat_Vmeans_gpu(hipStream_t stream = NULL);

	void stat_Imeans_gpu(hipStream_t stream = NULL);

	void update_Props_gpu(const unsigned int* neruon_indice,
							const unsigned int* prop_indice,
							const float* prop_vals,
							const unsigned int n,
							hipStream_t stream = NULL);

	void update_Prop_Cols_gpu(const unsigned int* prop_indice,
							const unsigned int* brain_indice,
							const T* hp_vals,
							const unsigned int n,
							hipStream_t stream = NULL);

	void update_Gamma_Prop_Cols_gpu(const unsigned int* prop_indice,
											const unsigned int* brain_indice,
											const T* alphas,
											const T* betas,
											const unsigned int n,
											hipStream_t stream = NULL);

	void stat_Spikes_and_Vmembs_gpu(const unsigned int* samples,
											const unsigned int n,
											char* spikes,
											T* vmembs,
											hipStream_t stream = NULL);

	void record_F_sending_actives(const map<unsigned short, tuple<unsigned int*, int>>& rank_map);

	void update_F_routing_spikes_gpu(const unsigned int* sending_colinds,
											const unsigned int sending_num,
											const unsigned int f_active_num,
											unsigned char* f_actives,
											hipStream_t stream = NULL);
	void update_F_routing_actives_gpu(const unsigned char* f_actives,
											const unsigned int* sending_rowptrs,
											const unsigned int* sending_colinds,
											const unsigned int segments,
											const unsigned int sending_count,
											unsigned char* f_sending_actives,
											unsigned int* block_rowptrs,
											unsigned int* active_rowptrs,
											unsigned int* active_colinds,
											hipStream_t stream = NULL);

	void count_F_sending_actives_temporary_storage_size(const unsigned int sending_count,
																const unsigned int segments,
																unsigned int* block_rowptrs,
																unsigned int* active_rowptrs,
																size_t& storage_size_bytes,
																hipStream_t stream = NULL);
	
	void update_F_sending_actives_gpu(hipStream_t stream = NULL);

	void update_F_routing_offsets_gpu(const unsigned int* d_unions,
											const unsigned int union_num,
											const unsigned int offset_num,
											unsigned int* d_offsets,
											hipStream_t stream = NULL);

	void update_F_routing_neuron_ids_gpu(const unsigned int* d_unions,
												const unsigned int colind_num,
												unsigned int* d_colinds,
												hipStream_t stream = NULL);

	void update_F_recving_actives_gpu(hipStream_t stream = NULL);

	void add_sample_neurons_gpu(unsigned int* samples,
								const unsigned int n);
	void update_sample_neurons_gpu(hipStream_t stream = NULL);

/*
	void stat_MaxErr_V_membrane_for_spiking_neurons_gpu(hipStream_t stream = NULL);

	void stat_Mean_I_synaptic_gpu(hipStream_t stream = NULL);

	void stat_Variance_I_synaptic_gpu(hipStream_t stream = NULL);
*/	
	void init_all_params_and_stages_cpu();
	void update_I_synaptic_cpu();
	void update_I_ext_stimulus_cpu()
	{
		if(nullptr != i_ext_stimuli_)
		{
			hipMemcpy(i_ext_stimuli_->mutable_cpu_data(), i_ext_stimuli_->gpu_data(), i_ext_stimuli_->size(), hipMemcpyDeviceToHost);
		}
	}
	void update_V_membrane_cpu();
	void update_F_active_cpu();

	void update_F_sending_actives_cpu();
	
	void update_J_presynaptic_inner_cpu();
	void update_J_presynaptic_outer_cpu();
	void update_J_presynaptic_cpu();

	void reset_V_membrane_cpu();

	void stat_Freqs_cpu();
	void stat_Vmeans_and_Imeans_cpu(bool has_vmean,
											bool has_imean);

	void stat_Spikes_and_Vmembs_cpu(const unsigned int* samples,
											const unsigned int n,
											char* spikes,
											T* vmembs);
	
	void update_sample_neurons_cpu();
	
	void update_F_routing_spikes_cpu(const unsigned int* sending_colinds,
											const unsigned int sending_num,
											const unsigned int n,
											unsigned char* f_actives);

	void update_F_routing_actives_cpu(const unsigned char* f_actives,
											const unsigned int* sending_rowptrs,
											const unsigned int* sending_colinds,
											const unsigned int segments,
											unsigned int* f_active_rowptrs,
											unsigned int* f_active_colinds);

	void update_F_routing_offsets_cpu(const unsigned int* unions,
											const unsigned int union_num,
											const unsigned int rows,
											const unsigned int* rowptrs,
											unsigned int* colinds);

	void update_F_routing_neuron_ids_cpu(const unsigned int* unions,
											const unsigned int colind_num,
											unsigned int* colinds);

	void update_F_recving_actives_cpu();


	void set_time(const T t)
	{
		t_ = t;
	}
	
	void update_time()
	{
		t_ += delta_t_;
	}

	const unsigned long long get_seed() const
	{
		return seed_;
	}

	const unsigned int get_input_timestamp_size() const
	{
		if(nullptr != input_timestamps_)
		{
			assert(BLOCK_TYPE_INPUT == block_type_);
			return input_timestamps_->count();
		}
		else
		{
			return 0;
		}
	}
	
	const unsigned int* get_input_timestamp_data() const
	{
		if(nullptr != input_timestamps_)
                {
                        assert(BLOCK_TYPE_INPUT == block_type_);	
			return input_timestamps_->cpu_data();
		}
		else
		{
			return NULL;
		}
	}

	unsigned short get_block_id() const
	{
		return bid_;
	}
	
	unsigned int get_total_neurons() const
	{
		return total_neurons_;
	}

	const unsigned char* get_F_actives_gpu() const
	{
		return f_inner_actives_->gpu_data();
	}

	const unsigned char* get_F_actives_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(f_inner_actives_->mutable_cpu_data(), f_inner_actives_->gpu_data(), f_inner_actives_->size(), hipMemcpyDeviceToHost));
		return f_inner_actives_->cpu_data();
	}

	unsigned char* get_mutable_F_actives_gpu()
	{
		return f_inner_actives_->mutable_gpu_data();
	}
	
	unsigned char* get_mutable_F_actives_cpu()
	{
		return f_inner_actives_->mutable_cpu_data();
	}

	const unsigned char* get_F_actives_without_noise_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(f_inner_actives_without_noise_->mutable_cpu_data(), f_inner_actives_->gpu_data(), f_inner_actives_without_noise_->size(), hipMemcpyDeviceToHost));
		return f_inner_actives_without_noise_->cpu_data();
	}
	
	const T* get_V_membranes_gpu() const
	{
		return v_membranes_->gpu_data();
	}

	const T* get_V_membranes_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(v_membranes_->mutable_cpu_data(), v_membranes_->gpu_data(), v_membranes_->size(), hipMemcpyDeviceToHost));
		return v_membranes_->cpu_data();
	}

	const T* get_T_actives_gpu() const
	{
		return t_actives_->gpu_data();
	}

	const T* get_T_actives_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(t_actives_->mutable_cpu_data(), t_actives_->gpu_data(), t_actives_->size(), hipMemcpyDeviceToHost));
		return t_actives_->cpu_data();
	}

	const T2* get_J_ex_presynaptics_gpu() const
	{
		return j_ex_presynaptics_->gpu_data();
	}

	const T2* get_J_ex_presynaptics_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(j_ex_presynaptics_->mutable_cpu_data(), j_ex_presynaptics_->gpu_data(), j_ex_presynaptics_->size(), hipMemcpyDeviceToHost));
		return j_ex_presynaptics_->cpu_data();
	}

	const T2* get_J_in_presynaptics_gpu() const
	{
		return j_in_presynaptics_->gpu_data();
	}

	const T2* get_J_in_presynaptics_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(j_in_presynaptics_->mutable_cpu_data(), j_in_presynaptics_->gpu_data(), j_in_presynaptics_->size(), hipMemcpyDeviceToHost));
		return j_in_presynaptics_->cpu_data();
	}

	void get_J_presynaptic_cpu(shared_ptr<DataAllocator<T>> j_presynaptics)
	{
		if(nullptr == j_presynaptics)
			return;
		transform_J_presynaptic_gpu(j_presynaptics->mutable_gpu_data());
		HIP_CHECK(hipMemcpy(j_presynaptics->mutable_cpu_data(), j_presynaptics->gpu_data(), j_presynaptics->size(), hipMemcpyDeviceToHost));
	}
	
	const T* get_I_synaptics_gpu() const
	{
		return i_synaptics_->gpu_data();
	}

	const T* get_I_synaptics_cpu(bool synced = true) const
	{
		if(synced)
			HIP_CHECK(hipMemcpy(i_synaptics_->mutable_cpu_data(), i_synaptics_->gpu_data(), i_synaptics_->size(), hipMemcpyDeviceToHost));
		return i_synaptics_->cpu_data();
	}

	const T* get_noise_rate_gpu() const
	{
		return noise_rates_->gpu_data();
	}

	const T* get_noise_rate_cpu(bool synced = true) const
	{
		if(synced)
			HIP_CHECK(hipMemcpy(noise_rates_->mutable_cpu_data(), noise_rates_->gpu_data(), noise_rates_->size(), hipMemcpyDeviceToHost));
		return noise_rates_->cpu_data();
	}

	unsigned int get_total_subblocks() const
	{
		if(nullptr != sub_binfos_)
			return static_cast<unsigned int>(sub_binfos_->count());
		else
			return 0;
	}

	const unsigned int get_sub_bids() const
	{
		return sub_bids_->count();
	}

	const unsigned int* get_sub_bids_cpu() const
	{
		return sub_bids_->cpu_data();
	}

	const unsigned int* get_sub_bcounts_cpu() const
	{
		return sub_bcounts_->cpu_data();
	}
	
	const uint2* get_sub_binfos_gpu() const
	{
		return sub_binfos_->gpu_data();
	}

	const uint2* get_sub_binfos_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(sub_binfos_->mutable_cpu_data(), sub_binfos_->gpu_data(), sub_binfos_->size(), hipMemcpyDeviceToHost));
		return sub_binfos_->cpu_data();
	}

	const unsigned int* get_freqs_gpu() const
	{
		return freqs_->gpu_data();
	}

	const unsigned int* get_freqs_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(freqs_->mutable_cpu_data(), freqs_->gpu_data(), freqs_->size(), hipMemcpyDeviceToHost));
		return freqs_->cpu_data();
	}

	const T* get_vmeans_gpu() const
	{
		return vmeans_->gpu_data();
	}

	const T* get_vmeans_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(vmeans_->mutable_cpu_data(), vmeans_->gpu_data(), vmeans_->size(), hipMemcpyDeviceToHost));
		return vmeans_->cpu_data();
	}

	const T* get_imeans_gpu() const
	{
		return imeans_->gpu_data();
	}

	const T* get_imeans_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(imeans_->mutable_cpu_data(), imeans_->gpu_data(), imeans_->size(), hipMemcpyDeviceToHost));
		return imeans_->cpu_data();
	}

	const size_t get_inner_conninds_size() const
	{
		if(nullptr == inner_conninds_)
			return 0;
		return inner_conninds_->count();
	}

	const unsigned int* get_inner_conninds_gpu() const
	{
		return inner_conninds_->gpu_data();
	}

	const unsigned int* get_inner_conninds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_conninds_->mutable_cpu_data(), inner_conninds_->gpu_data(), inner_conninds_->size(), hipMemcpyDeviceToHost));
		return inner_conninds_->cpu_data();
	}
	
	const T2* get_inner_w_synaptics_gpu() const
	{
		return inner_w_synaptics_->gpu_data();
	}

	const size_t get_inner_w_synaptics_size()
	{
		return inner_w_synaptics_->count();
	}

	const T2* get_inner_w_synaptics_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_w_synaptics_->mutable_cpu_data(), inner_w_synaptics_->gpu_data(), inner_w_synaptics_->size(), hipMemcpyDeviceToHost));
		return inner_w_synaptics_->cpu_data();
	}
	
	const unsigned int* get_inner_colinds_gpu() const
	{
		return inner_colinds_->gpu_data();
	}

	const unsigned int* get_inner_colinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_colinds_->mutable_cpu_data(), inner_colinds_->gpu_data(), inner_colinds_->size(), hipMemcpyDeviceToHost));
		return inner_colinds_->cpu_data();
	}
	
	const unsigned int* get_inner_rowptrs_gpu() const
	{
		return inner_rowptrs_->gpu_data();
	}

	const unsigned int* get_inner_rowptrs_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_rowptrs_->mutable_cpu_data(), inner_rowptrs_->gpu_data(), inner_rowptrs_->size(), hipMemcpyDeviceToHost));
		return inner_rowptrs_->cpu_data();
	}

	const unsigned char* get_inner_connkinds_gpu() const
	{
		return inner_connkinds_->gpu_data();
	}

	const unsigned char* get_inner_connkinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_connkinds_->mutable_cpu_data(), inner_connkinds_->gpu_data(), inner_connkinds_->size(), hipMemcpyDeviceToHost));
		return inner_connkinds_->cpu_data();
	}
	
	const T2* get_outer_w_synaptics_gpu() const
	{
		return outer_w_synaptics_->gpu_data();
	}

	const unsigned int get_outer_w_synaptics_size()
	{
		return outer_w_synaptics_->count();
	}

	const T2* get_outer_w_synaptics_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_w_synaptics_->mutable_cpu_data(), outer_w_synaptics_->gpu_data(), outer_w_synaptics_->size(), hipMemcpyDeviceToHost));
		return outer_w_synaptics_->cpu_data();
	}

	const unsigned int* get_outer_colinds_gpu() const
	{
		return outer_colinds_->gpu_data();
	}

	const unsigned int* get_outer_colinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_colinds_->mutable_cpu_data(), outer_colinds_->gpu_data(), outer_colinds_->size(), hipMemcpyDeviceToHost));
		return outer_colinds_->cpu_data();
	}
	
	const unsigned int* get_outer_rowptrs_gpu() const
	{
		return outer_rowptrs_->gpu_data();
	}

	const unsigned int* get_outer_rowptrs_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_rowptrs_->mutable_cpu_data(), outer_rowptrs_->gpu_data(), outer_rowptrs_->size(), hipMemcpyDeviceToHost));
		return outer_rowptrs_->cpu_data();
	}

	const unsigned char* get_outer_connkinds_gpu() const
	{
		return outer_connkinds_->gpu_data();
	}

	const unsigned char* get_outer_connkinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_connkinds_->mutable_cpu_data(), outer_connkinds_->gpu_data(), outer_connkinds_->size(), hipMemcpyDeviceToHost));
		return outer_connkinds_->cpu_data();
	}

	const T* get_uniform_samples_gpu() const
	{
		return uniform_samples_->gpu_data();
	}

	const T* get_uniform_samples_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(uniform_samples_->mutable_cpu_data(), uniform_samples_->gpu_data(), uniform_samples_->size(), hipMemcpyDeviceToHost));
		return uniform_samples_->cpu_data();
	}

	void fetch_props(const unsigned int* neuron_indice,
					const unsigned int* prop_indice,
					const unsigned int n,
					vector<T>& result);

	void fetch_prop_cols(const unsigned int* prop_indice,
						const unsigned int* brain_indice,
						const unsigned int n,
						vector<vector<T>>& result);

protected:
	void transform_J_presynaptic_gpu(T* j_presynaptics,
										hipStream_t stream = NULL);
	
	int gpu_id_;
	shared_ptr<DataAllocator<T>> noise_rates_;
	shared_ptr<DataAllocator<T2>> g_ex_conducts_;
	shared_ptr<DataAllocator<T2>> g_in_conducts_;
	shared_ptr<DataAllocator<T2>> v_ex_membranes_;
	shared_ptr<DataAllocator<T2>> v_in_membranes_;
	shared_ptr<DataAllocator<T2>> tao_ex_constants_;
	shared_ptr<DataAllocator<T2>> tao_in_constants_;
	shared_ptr<DataAllocator<T>> v_resets_;
	shared_ptr<DataAllocator<T>> v_thresholds_;
	shared_ptr<DataAllocator<T>> c_membrane_reciprocals_;
	shared_ptr<DataAllocator<T>> v_leakages_;
	shared_ptr<DataAllocator<T>> g_leakages_;
	shared_ptr<DataAllocator<T>> t_refs_;

	shared_ptr<DataAllocator<unsigned char>> f_exclusive_flags_;
	shared_ptr<DataAllocator<unsigned int>> f_exclusive_counts_;

	shared_ptr<DataAllocator<T2>> j_ex_presynaptics_;
	shared_ptr<DataAllocator<T2>> j_ex_presynaptic_deltas_;
	shared_ptr<DataAllocator<T2>> j_in_presynaptics_;
	shared_ptr<DataAllocator<T2>> j_in_presynaptic_deltas_;
	shared_ptr<DataAllocator<T>> v_membranes_;
	shared_ptr<DataAllocator<T>> t_actives_;
	shared_ptr<DataAllocator<T>> i_synaptics_;
	shared_ptr<DataAllocator<T>> i_ext_stimuli_;
	
	unsigned long long seed_;
	std::atomic<long long> philox_seed_offset_;
	shared_ptr<DataAllocator<hiprandStateMtgp32>> gen_states_;
	shared_ptr<DataAllocator<mtgp32_kernel_params_t>> kernel_params_;
	
	shared_ptr<DataAllocator<T>> uniform_samples_;
	
	//input block spike
	shared_ptr<DataAllocator<unsigned int>> input_timestamps_;
	shared_ptr<DataAllocator<unsigned int>> input_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> input_colinds_;
	
	//intra block connecting spike
	shared_ptr<DataAllocator<unsigned char>> f_inner_actives_;
	shared_ptr<DataAllocator<unsigned char>> f_inner_actives_without_noise_;

	shared_ptr<DataAllocator<unsigned int>> inner_conninds_;
	//intra block connecting weight table in csr format
	shared_ptr<DataAllocator<T2>> inner_w_synaptics_;
	//Points to the integer array that contains the row indices 
	//of the corresponding nonzero elements in array csr_inner_w_synaptics_
	shared_ptr<DataAllocator<unsigned int>> inner_colinds_;
	//Points to the integer array of length n+1 (n rows in the sparse matrix)
	shared_ptr<DataAllocator<unsigned int>> inner_rowptrs_;
	shared_ptr<DataAllocator<unsigned char>> inner_connkinds_;

	//inter block connecting spike
	//shared_ptr<DataAllocator<unsigned char>> f_outer_actives_;

	//inter block connecting weight table in csr format
	shared_ptr<DataAllocator<T2>> outer_w_synaptics_;
	//Points to the integer array that contains the row indices 
	//of the corresponding nonzero elements in array csr_outer_w_synaptics_
	shared_ptr<DataAllocator<unsigned int>> outer_colinds_;
	//Points to the integer array of length n+1 (n rows in the sparse matrix)
	shared_ptr<DataAllocator<unsigned int>> outer_rowptrs_;
	shared_ptr<DataAllocator<unsigned char>> outer_connkinds_;

	unsigned int total_neurons_;  // numbers of neurons in this brain block
	unsigned short bid_;  // block id
	T t_;
	T delta_t_;

	shared_ptr<DataAllocator<unsigned int>> sub_bids_;
	shared_ptr<DataAllocator<unsigned int>> sub_bcounts_;
	shared_ptr<DataAllocator<uint2>> sub_binfos_;
	shared_ptr<DataAllocator<unsigned int>> freqs_;
	shared_ptr<DataAllocator<T>> vmeans_;
	shared_ptr<DataAllocator<T>> imeans_;

public:
	BLOCK_TYPE block_type_;
	vector<unsigned short> f_sending_bids_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_colinds_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_active_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>>	f_sending_active_colinds_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_block_rowptrs_;
	shared_ptr<DataAllocator<unsigned char>> f_sending_actives_;
	
	vector<unsigned short> f_receiving_bids_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_colinds_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_active_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_active_colinds_;
	
	shared_ptr<DataAllocator<unsigned int>> sample_indices_;
	shared_ptr<DataAllocator<unsigned char>> f_sample_actives_;
	shared_ptr<DataAllocator<T>> v_sample_membranes_;
	
private:
	DISABLE_COPY_AND_ASSIGN(BrainBlock);
};

}//namespace dtb
