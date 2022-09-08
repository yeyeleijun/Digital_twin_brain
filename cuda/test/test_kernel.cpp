#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <chrono>
#include "common.hpp"
#include "util/cmd_arg.hpp"
#include "util/cnpy.h"
#include "block.hpp"
#include <cuda_profiler_api.h>

using namespace std;
using namespace istbi;
using chrono::steady_clock;
using chrono::time_point;
using chrono::duration;


int main(int argc, char **argv)
{
	float delta_t = 1.f, noise_rate = .0f;
	unsigned int bid = 0, gid = 0;
	string filename;
	
	
	const char* path = "/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single10m/block_all";
	/*
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
	*/
	CUDA_CHECK(cudaSetDevice(gid));
	// Clear error status
  	CUDA_CHECK(cudaGetLastError());
	cout << "initialize brain block ..." << endl;
	{
		unique_ptr<BrainBlock<float, float2>> block = nullptr;
		{
			filename = string(path) + string("/block_") + to_string(bid) + string(".npz");
			cnpy::npz_t arr = cnpy::npz_load(filename);

			cout << "load npz file ..." << endl;
			cnpy::NpyArray arr_prop = arr["property"];
			float* props = arr_prop.data<float>();
			assert(arr_prop.shape.size() == 2 && arr_prop.shape[1] == 22);

			cnpy::NpyArray arr_nids = arr["output_neuron_idx"];
			unsigned int* nids = arr_nids.data<unsigned int>();
			assert(arr_nids.shape.size() == 1);

			cnpy::NpyArray arr_conn_bids = arr["input_block_idx"];
			unsigned short* conn_bids = arr_conn_bids.data<unsigned short>();
			assert(arr_conn_bids.shape.size() == 1);

			cnpy::NpyArray arr_conn_nids = arr["input_neuron_idx"];
			unsigned int* conn_nids = arr_conn_nids.data<unsigned int>();
			assert(arr_conn_nids.shape.size() == 1);

			cnpy::NpyArray arr_conn_kinds = arr["input_channel_offset"];
			unsigned char* conn_kinds = arr_conn_kinds.data<unsigned char>();
			assert(arr_conn_kinds.shape.size() == 1);

			assert(arr_nids.shape[0] == arr_conn_bids.shape[0] &&
				arr_conn_bids.shape[0] == arr_conn_nids.shape[0] &&
				arr_conn_nids.shape[0] == arr_conn_kinds.shape[0]);

			cnpy::NpyArray arr_weight = arr["weight"];
			float2* weights = reinterpret_cast<float2*>(arr_weight.data<float>());
			assert(arr_weight.shape.size() == 2 &&
				arr_weight.shape[0] == arr_nids.shape[0] &&
				arr_weight.shape[1] == 2);
			
			unique_ptr<BrainBlock<float, float2>> p(new BrainBlock<float, float2>(arr_prop.shape[0], bid, gid, delta_t));
			p->init_connection_table_gpu(nids,
										conn_bids,
										conn_nids,
										conn_kinds,
										weights,
										arr_nids.shape[0]);

		
			p->init_config_params_gpu(props, arr_prop.shape[0], arr_prop.shape[1]);
			p->init_all_stages_gpu();
			p->reset_V_membrane_gpu();
			CUDA_CHECK(cudaDeviceSynchronize());
			block = std::move(p);
		}

		double max_duration, min_duration, avg_duration, tmp_duration;
		time_point<steady_clock> start, end;
		duration<double> diff;
		std::cout << "iteration begining ..." << std::endl;
		for(int i = 0; i < 1; i++)
		{
			start = steady_clock::now();
			
			block->update_time();
			//cudaProfilerStart();
			block->update_V_membrane_gpu();
			//block->stat_Freqs_and_Vmeans_gpu();	
			block->update_F_active_gpu(noise_rate);
			block->update_J_presynaptic_inner_gpu();
			block->update_J_presynaptic_outer_gpu();
			block->update_J_presynaptic_gpu();
			block->update_I_synaptic_gpu();
			
			CUDA_CHECK(cudaDeviceSynchronize());
			std::cout << "iteration " << i << " done..." << std::endl;
			end = steady_clock::now();
			diff = end - start;
			tmp_duration = diff.count();
			if(i == 0)
			{
				max_duration = tmp_duration;
				min_duration = tmp_duration;
				avg_duration = tmp_duration;
			}
			else
			{
				max_duration = MAX(max_duration, tmp_duration);
				min_duration = MIN(min_duration, tmp_duration);
				avg_duration += tmp_duration;
			}
		//cudaProfilerStop();
		}

		avg_duration /= 800;
		std::cout << "max duration\tmin duration\tavg duration" << std::endl;
		std::cout << max_duration << "\t" << min_duration << "\t" << avg_duration << std::endl;
	}
	
	DEVICE_RESET
	return 0;
}
