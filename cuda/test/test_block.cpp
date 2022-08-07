#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include "common.hpp"
#include "test/check.hpp"
#include "test/save_result.hpp"
#include "util/cmd_arg.hpp"
#include "util/cnpy.h"

using namespace std;
using namespace istbi;

template<typename T>
static void resort_result(shared_ptr<DataAllocator<long long>>& resorting_buff,
						const T* d_keys,
						const unsigned int n,
						shared_ptr<DataAllocator<T>>& saving_result)
{
	resort_result_gpu<T>(resorting_buff->gpu_data(),
						 d_keys, 
						 n,
						 saving_result->mutable_gpu_data());
	CUDA_CHECK(cudaMemcpy(saving_result->mutable_cpu_data(), saving_result->gpu_data(), saving_result->size(), cudaMemcpyDeviceToHost));
}

int main(int argc, char **argv)
{
	float delta_t = 1.f;
	unsigned int total_neurons;
	string filename;
	ostream* os;
	
	char* path = NULL;
	if(!get_cmdline_argstring(argc, (const char**)argv, "fp", &path))
	{
		cerr << "no specified exec path" << endl;
		return -1;
	}

	unsigned int bid = get_cmdline_argint(argc, (const char**)argv, "bid");
	unsigned int gid = get_cmdline_argint(argc, (const char**)argv, "gid");
	
	float noise_rate = get_cmdline_argfloat(argc, (const char**)argv, "nr");
	if(noise_rate == (float)0)
		noise_rate = .01f;
	
	int iters = get_cmdline_argint(argc, (const char**)argv, "it");
	if(iters == 0)
		iters = 4000;
	
	int preset_sample = get_cmdline_argint(argc, (const char**)argv, "ps");
	int saving_gpu = get_cmdline_argint(argc, (const char**)argv, "sg");
	int check_cpu = get_cmdline_argint(argc, (const char**)argv, "cc");
	int check_param = get_cmdline_argint(argc, (const char**)argv, "cp");
	int python_check = get_cmdline_argint(argc, (const char**)argv, "pc");

	int saving_log = get_cmdline_argint(argc, (const char**)argv, "sl");

	if(saving_log)
	{
		filename = string(path) + string("/output.log"); 
		os = new std::ofstream(filename.c_str());
	}
	else
	{
		os = &cout;
	}
	
	CUDA_CHECK(cudaSetDevice(gid));
	// Clear error status
  	CUDA_CHECK(cudaGetLastError());

	{
		vector<shared_ptr<DataAllocator<float>>> sample_vect;
		shared_ptr<DataAllocator<float2>> saving_weight;
		shared_ptr<DataAllocator<float>> saving_data;
		shared_ptr<DataAllocator<unsigned char>> saving_buff;
		shared_ptr<DataAllocator<long long>> resorting_buff;
		shared_ptr<DataAllocator<unsigned char>> saving_flags;
		
		filename = string(path) + string("/block_") + to_string(bid) + string(".npz");

		shared_ptr<BrainBlock<float, float2>> shared_block = init_brain_block<float, float2>(filename.c_str(), delta_t, bid, gid);
		BrainBlock<float, float2>* block = shared_block.get();
		total_neurons = block->get_total_neurons();

		if(check_param)
		{
			(*os) << "check address table parameters..." << endl;
			check_params<float, float2>(block, filename.c_str());
			(*os) << "PASS" << endl;
		}
		
		if(saving_gpu)
		{
			filename = string(path) + string("/resort_idx.npy");
			cnpy::NpyArray arr = cnpy::npy_load(filename);
			long long* resort_inds = arr.data<long long>();
			assert(arr.shape.size() == 1 && arr.shape[0] == total_neurons);
			resorting_buff = make_shared<DataAllocator<long long>>(sizeof(long long) * total_neurons);
			saving_weight = make_shared<DataAllocator<float2>>(sizeof(float2) * total_neurons);
			saving_data = make_shared<DataAllocator<float>>(sizeof(float) * total_neurons);
			saving_buff = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * total_neurons);
			saving_flags = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * total_neurons);
			CUDA_CHECK(cudaMemcpy(resorting_buff->mutable_gpu_data(), resort_inds, resorting_buff->size(), cudaMemcpyHostToDevice));
		}
		
		block->init_all_stages_gpu();
		CUDA_CHECK(cudaDeviceSynchronize());

		if(check_cpu)
		{
			block->init_all_params_and_stages_cpu();
			block->update_I_synaptic_cpu();
			block->reset_V_membrane_cpu();
		}
		
		block->reset_V_membrane_gpu();
		CUDA_CHECK(cudaDeviceSynchronize());

		if(check_cpu)
		{
			check_result<float, float2>(block, 0, *os);
		}

		if(preset_sample)
		{	
			filename = path + string("/sample.npy");
			read_samples_from_preset<float>(filename.c_str(), sample_vect);
		}
			
		for(int i = 0; i < iters; i++)
		{
			block->update_time();
			block->update_V_membrane_gpu();

			block->stat_Freqs_and_Vmeans_gpu();
			int sidx; 
			if(preset_sample)
			{
				sidx = static_cast<int>(i % sample_vect.size());
				block->update_F_active_from_preset_gpu(sample_vect[sidx]->cpu_data(), sample_vect[sidx]->count(), noise_rate);
			}
			else
			{
				if(check_cpu)
					block->update_F_active_gpu(noise_rate, 0, 1, true);
				else
					block->update_F_active_gpu(noise_rate);
			}
			
			block->update_J_presynaptic_inner_gpu();
			block->update_J_presynaptic_outer_gpu();
			if(check_cpu)
				block->update_J_presynaptic_gpu(true);
			else
				block->update_J_presynaptic_gpu();
			
			block->update_I_synaptic_gpu();

			if(saving_gpu)
				save_spike_gpu(block->get_F_actives_gpu(),
							   total_neurons,
							   saving_flags->mutable_gpu_data());
			CUDA_CHECK(cudaDeviceSynchronize());
			
			if(check_cpu)
			{
				block->update_V_membrane_cpu();
				if(preset_sample)
					block->update_F_active_from_preset_cpu(sample_vect[sidx]->cpu_data(), sample_vect[sidx]->count(), noise_rate);
				else
					block->update_F_active_cpu(noise_rate);
				
				block->update_J_presynaptic_inner_cpu();
				block->update_J_presynaptic_outer_cpu();
				block->update_J_presynaptic_cpu();
				block->update_I_synaptic_cpu();
				(*os) << "========gpu vs. cpu====================" << endl;
				check_result<float, float2>(block, i + 1, *os);
			}

			if(python_check)
			{
				(*os) << "========gpu vs. python====================" << endl;
				filename = path + string("/result_") + to_string(i) + string(".npz");
				check_result<float, float2>(block, filename.c_str(), i + 1, *os);
			}

			if(saving_gpu)
			{
				vector<size_t> shape(2);
				shape[0] = 1;
				shape[1] = 2 * total_neurons;
				if(i == 0)
				{

					filename = path + string("/ex_presynaptic_0.npy");
					resort_result<float2>(resorting_buff,
										  block->get_J_ex_presynaptics_gpu(),
										  total_neurons,
										  saving_weight);
					cnpy::npy_save<float>(filename, reinterpret_cast<const float*>(saving_weight->cpu_data()), shape, "w");

					filename = path + string("/in_presynaptic_0.npy");
					resort_result<float2>(resorting_buff,
										  block->get_J_in_presynaptics_gpu(),
										  total_neurons,
										  saving_weight);
					cnpy::npy_save<float>(filename, reinterpret_cast<const float*>(saving_weight->cpu_data()), shape, "w");

					shape[1] = total_neurons;

					filename = path + string("/membrane_0.npy");
					resort_result<float>(resorting_buff,
										  block->get_V_membranes_gpu(),
										  total_neurons,
										  saving_data);
					cnpy::npy_save<float>(filename, saving_data->cpu_data(), shape, "w");

					filename = path + string("/synaptic_0.npy");
					resort_result<float>(resorting_buff,
										  block->get_I_synaptics_gpu(),
										  total_neurons,
										  saving_data);
					cnpy::npy_save<float>(filename, saving_data->cpu_data(), shape, "w");

					filename = path + string("/flag_0.npy");
					resort_result<unsigned char>(resorting_buff,
										  saving_flags->gpu_data(),
										  total_neurons,
										  saving_buff);
					
					cnpy::npy_save<unsigned char>(filename, saving_buff->cpu_data(), shape, "w");
				}
				else
				{
					filename = path + string("/ex_presynaptic_0.npy");
					resort_result<float2>(resorting_buff,
										  block->get_J_ex_presynaptics_gpu(),
										  total_neurons,
										  saving_weight);
					cnpy::npy_save<float>(filename, reinterpret_cast<const float*>(saving_weight->cpu_data()), shape, "a");

					filename = path + string("/in_presynaptic_0.npy");
					resort_result<float2>(resorting_buff,
										  block->get_J_in_presynaptics_gpu(),
										  total_neurons,
										  saving_weight);
					cnpy::npy_save<float>(filename, reinterpret_cast<const float*>(saving_weight->cpu_data()), shape, "a");

					shape[1] = total_neurons;

					filename = path + string("/membrane_0.npy");
					resort_result<float>(resorting_buff,
										  block->get_V_membranes_gpu(),
										  total_neurons,
										  saving_data);
					cnpy::npy_save<float>(filename, saving_data->cpu_data(), shape, "a");

					filename = path + string("/synaptic_0.npy");
					resort_result<float>(resorting_buff,
										  block->get_I_synaptics_gpu(),
										  total_neurons,
										  saving_data);
					cnpy::npy_save<float>(filename, saving_data->cpu_data(), shape, "a");

					filename = path + string("/flag_0.npy");
					resort_result<unsigned char>(resorting_buff,
										  saving_flags->gpu_data(),
										  total_neurons,
										  saving_buff);
					
					cnpy::npy_save<unsigned char>(filename, saving_buff->cpu_data(), shape, "a");
				}
			}
		}
	}
	
	if(os != &cout) 
    	delete os;
	
	DEVICE_RESET
	return 0;
}
