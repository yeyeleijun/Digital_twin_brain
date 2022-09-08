#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cassert>
#include <mpi.h>
#include <unistd.h>
#include "common.hpp"
#include "block.hpp"
#include "util/cnpy.h"
#include "test/check.hpp"
#include "test/save_result.hpp"
#include "util/cmd_arg.hpp"

#define ENV_LOCAL_RANK 		"OMPI_COMM_WORLD_LOCAL_RANK"
#define MPI_MASTER_RANK		0

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   					\
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

using namespace std;
using namespace istbi;

static void init_mpi_env(int* argc, char*** argv, int* rank, int& gpu_id, int* size)
{
	// Setting the device here will have an effect only for the CUDA-aware MPI
	char* local_rank_str = NULL;
	int dev_count = 0;

	// We extract the local rank initialization using an environment variable
	if((local_rank_str = getenv(ENV_LOCAL_RANK)) != NULL)
	{
		*rank = atoi(local_rank_str);
	}

	
	CUDA_CHECK(cudaGetDeviceCount(&dev_count));
	gpu_id = (*rank) % dev_count;
	CUDA_CHECK(cudaSetDevice(gpu_id));
	
	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, size);

	char hostname[1024];
	gethostname(hostname, 1024);
	cout << "The rank (" << *rank << ") within the node " << hostname << "." << endl;
}

static void pre_run_snn(BrainBlock<float, float2>* block, MPI_Comm comm, int rank, int size, int tag)
{
	struct cudaDeviceProp devProps;
	int dev = 0, enabledECC = 0;

	// We get the properties of the current device, assuming all other devices are the same
	CUDA_CHECK(cudaGetDevice(&dev));
	CUDA_CHECK(cudaGetDeviceProperties(&devProps, dev));

	// Determine how many devices have ECC enabled (assuming exactly one process per device)
	MPI_Reduce(&devProps.ECCEnabled, &enabledECC, 1, MPI_INT, MPI_SUM, MPI_MASTER_RANK, comm);
	MPI_Barrier(comm);
	
	if(rank == MPI_MASTER_RANK)
		cout << "Starting SNN with " << size <<" processes using " << devProps.name << " GPUs (ECC enabled: " << enabledECC << " / " << size << "):" << endl;

	vector<MPI_Request> requests(size);
	vector<MPI_Status> status(size);
	{
		unsigned int last_idx = 0;
		unsigned short* bids = NULL;
		unsigned int* inds = NULL;
		unsigned int* nids = NULL;
		if(!block->f_receiving_ranks_.empty())
		{
			bids = block->f_receiving_ranks_.data();
			inds = block->f_receiving_rowptrs_->mutable_cpu_data();
			nids = block->f_receiving_colinds_->mutable_cpu_data();
			assert((block->f_receiving_rowptrs_->count() == (block->f_receiving_ranks_.size() + 1)) && 
				(block->f_receiving_ranks_.size() < size));
		}
		
		for(unsigned int i = 0; i < size; i++)
		{
			if(i == rank)
			{
				if(!block->f_receiving_ranks_.empty())
				{
					for(unsigned int j = 0; j < block->f_receiving_ranks_.size(); j++)
					{
						assert((bids[j] < size) &&
							(bids[j] != rank));
					}
				}
				
				continue;
			}

			if(!block->f_receiving_ranks_.empty())
			{
				if(last_idx < block->f_receiving_ranks_.size() && 
					i == bids[last_idx])
				{
					unsigned int elems = inds[last_idx + 1] - inds[last_idx];
					MPI_Isend(nids + inds[last_idx], elems, MPI_UNSIGNED, i, tag, comm, &requests[i]);
					last_idx++;
				}
				else
				{
					unsigned int data = ((unsigned int)-1);
					MPI_Isend(&data, 1, MPI_UNSIGNED, i, tag, comm, &requests[i]);
				}
			}
			else
			{
				unsigned int data = ((unsigned int)-1);
				MPI_Isend(&data, 1, MPI_UNSIGNED, i, tag, comm, &requests[i]);
			}
		
		}
	}
	
	{
		map<unsigned short, shared_ptr<DataAllocator<unsigned int>>> sending_f_map;
		const unsigned int bufsize = block->get_total_neurons() * 2;
		vector<unsigned int> buff(bufsize);
		for(unsigned int i = 0; i < size; i++)
		{
			if(i == rank)
				continue;
			
			MPI_Recv(buff.data(), bufsize, MPI_UNSIGNED, i, tag, comm, &status[i]);
			int elems;
			MPI_Get_count(&status[i], MPI_UNSIGNED, &elems);
			if((1 == elems) && (buff[0] == ((unsigned int)-1)))
				continue;

			assert(sending_f_map.find(i) == sending_f_map.end());
			sending_f_map.emplace(static_cast<unsigned short>(i), make_shared<DataAllocator<unsigned int>>(elems * sizeof(unsigned int)));
			memcpy((sending_f_map[i])->mutable_cpu_data(), buff.data(), (sending_f_map[i])->size());
		}
		
		block->record_F_sending_actives(sending_f_map);
	}

	for(unsigned int i = 0; i < size; i++)
	{
		if(i == rank)
			continue;
		
		int err = MPI_Wait(&requests[i], MPI_STATUSES_IGNORE);
		assert(err == MPI_SUCCESS);
	}
	
	MPI_Barrier(comm);
}


int main(int argc, char **argv)
{
	float delta_t = 1.f;
	int gid;
	unsigned int total_neurons;
	string filename;
	ostream* os;
	static const char* selection_file = "sample_selection_idx.npy";
	
	char* path;
	char* rpath;
	char* sname;
	if(!get_cmdline_argstring(argc, (const char**)argv, "fp", &path))
	{
		cerr << "no specified exec path" << endl;
		return -1;
	}

	if(!get_cmdline_argstring(argc, (const char**)argv, "rp", &rpath))
	{
		rpath = path;
	}

	if(!get_cmdline_argstring(argc, (const char**)argv, "sn", &sname))
	{
		sname = const_cast<char*>(selection_file);
	}

	int preset_sample = get_cmdline_argint(argc, (const char**)argv, "ps");

	int check_cpu = get_cmdline_argint(argc, (const char**)argv, "cc");

	int check_param = get_cmdline_argint(argc, (const char**)argv, "cp");
	
	int saving_gpu = get_cmdline_argint(argc, (const char**)argv, "sg");

	int saving_sample = get_cmdline_argint(argc, (const char**)argv, "ss");

	int saving_exchange = get_cmdline_argint(argc, (const char**)argv, "se");

	int saving_number = get_cmdline_argint(argc, (const char**)argv, "sb");

	float noise_rate = get_cmdline_argfloat(argc, (const char**)argv, "nr");
	if(noise_rate == (float)0)
		noise_rate = .01f;
	
	int iters = get_cmdline_argint(argc, (const char**)argv, "it");
	if(iters == 0)
		iters = 4000;

	int saving_log = get_cmdline_argint(argc, (const char**)argv, "sl");

	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	int tag = 0;
	//double time_start;
	init_mpi_env(&argc, &argv, &rank, gid, &size);

	if(saving_log)
	{
		filename = string(rpath) + string("/output_") + to_string(rank) + string(".log"); 
		os = new ofstream(filename.c_str());
	}
	else
	{
		os = &cout;
	}

	if(rank == MPI_MASTER_RANK)
		(*os) << "noise rate:" << noise_rate << endl;
	
	// Clear error status
  	CUDA_CHECK(cudaGetLastError());
	{
		vector<shared_ptr<DataAllocator<float>>> sample_vect;
		//shared_ptr<DataAllocator<unsigned char>> f_sending_actives;
		shared_ptr<DataAllocator<unsigned char>> sending_buff;
		shared_ptr<DataAllocator<unsigned char>> native_send_buff;
		//shared_ptr<DataAllocator<unsigned char>> f_receiving_actives;
		shared_ptr<DataAllocator<unsigned char>> receiving_buff;
		shared_ptr<DataAllocator<unsigned char>> calibrated_recv_buff;
		shared_ptr<DataAllocator<unsigned char>> saving_buff;

		vector<unsigned char> sending_buff_cpu;
		vector<unsigned char> receiving_buff_cpu;

		shared_ptr<CudaStream> stream1 = CudaStream::create();
		shared_ptr<CudaStream> stream2 = CudaStream::create();
		
		filename = string(path) + string("/block_") + to_string(rank) + string(".npz");
		shared_ptr<BrainBlock<float, float2>> shared_block = init_brain_block<float, float2>(filename.c_str(), delta_t, rank, gid);
		BrainBlock<float, float2>* block = shared_block.get();
		total_neurons = block->get_total_neurons();

		//(*os) << "random seed: " << block->get_seed() << " at the rank (" << rank << ")" << endl;
		
		if(check_param)
		{
			(*os) << "check address table parameters..." << endl;
			check_params<float, float2>(block, filename.c_str());
			(*os) << "PASS" << endl;
		}

		if(saving_sample)
		{
			filename = string(path) + string("/") + string(sname);
			cnpy::NpyArray arr = cnpy::npy_load(filename); 
			long long* samples = arr.data<long long>();
			assert(arr.word_size == sizeof(long long) && arr.shape.size() == 2);
			vector<unsigned int> sample_indices;
			{
				vector<unsigned int> sample_indices_cpu;
				resort_samples_gpu(samples, arr.shape[0], arr.shape[1], rank, sample_indices);
				resort_samples_cpu(samples, arr.shape[0], arr.shape[1], rank, sample_indices_cpu);
				assert(sample_indices.size() == sample_indices_cpu.size());
				for(unsigned int i = 0; i < sample_indices.size(); i++)
				{
					assert(sample_indices[i] == sample_indices_cpu[i]);
				}
			}
			
			if(!sample_indices.empty())
			{
				block->add_sample_neurons_gpu(sample_indices.data(), sample_indices.size());
			}
		}
		
		saving_buff = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * total_neurons); 

		if(preset_sample)
		{
			filename = string(path) + string("/sample_") + to_string(rank) + string(".npy");
			read_samples_from_preset<float>(filename.c_str(), sample_vect);
		}

		pre_run_snn(block, comm, rank, size, tag);
		
		#ifdef MPI_DEBUG
		cout << "=================================" << endl;
		int gdb_break = 1;
		while(gdb_break) {};
		#endif
		
		block->init_all_stages_gpu();
		//f_receiving_actives = make_shared<DataAllocator<unsigned char>>(block->f_receiving_actives_->size());
		CUDA_CHECK(cudaDeviceSynchronize());
		
		if(check_cpu)
		{
			block->init_all_params_and_stages_cpu();
			block->update_I_synaptic_cpu();
			block->reset_V_membrane_cpu();
		}
		
		block->reset_V_membrane_gpu();
		CUDA_CHECK(cudaDeviceSynchronize());

		vector<MPI_Request> requests;
		vector<MPI_Status> status;
		vector<unsigned int> send_bytes;
		vector<unsigned int> recv_bytes;
		if(!block->f_sending_ranks_.empty())
		{
			requests.resize(block->f_sending_ranks_.size());
			//unsigned int m = 0;
			for(unsigned int i = 0; i < block->f_sending_ranks_.size(); i++)
			{
				unsigned int elems = (block->f_sending_rowptrs_->cpu_data())[i + 1] - (block->f_sending_rowptrs_->cpu_data())[i];
				#if 0
				assert(block->f_sending_offsets_->cpu_data()[i] == m);
				m += (align_up<5, unsigned int>(elems) >> 3);
				if(rank == 0)
				{
					unsigned int j = block->f_sending_ranks_[i];
					switch(j)
					{
						case 1:
						case 2:
							assert(elems == 402701);
							break;
						case 3:
							assert(elems == 402705);
							break;
						default:
							assert(0);
					}
				}
				else if(rank == 1)
				{
					switch(block->f_sending_ranks_[i])
					{
						case 0:
							assert(elems == 402701);
							break;
						case 2:
							assert(elems == 398257);
							break;
						case 3:
							assert(elems == 398256);
							break;
						default:
							assert(0);
					}
				}
				else if(rank == 2)
				{
					switch(block->f_sending_ranks_[i])
					{
						case 0:
							assert(elems == 402702);
							break;
						case 1:
							assert(elems == 405144);
							break;
						case 3:
							assert(elems == 402698);
							break;
						default:
							assert(0);
					}
				}
				else if(rank == 3)
				{
					switch(block->f_sending_ranks_[i])
					{
						case 0:
							assert(elems == 398254);
							break;
						case 1:
							assert(elems == 398254);
							break;
						case 2:
							assert(elems == 402699);
							break;
						default:
							assert(0);
					}
				}
				else
					assert(0);
				#endif
				elems = (align_up<3, unsigned int>(elems) >> 3);
				send_bytes.push_back(elems);
			}

			if(check_cpu)
				sending_buff_cpu.resize((block->f_sending_rowptrs_->cpu_data())[block->f_sending_ranks_.size()]);

			//f_sending_actives = make_shared<DataAllocator<unsigned char>>(block->f_sending_actives_->size());
			if(saving_exchange)
			{
				native_send_buff = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * (block->f_sending_rowptrs_->cpu_data())[block->f_sending_ranks_.size()]);
				sending_buff = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * (block->f_sending_rowptrs_->cpu_data())[block->f_sending_ranks_.size()]); 
				
				CUDA_CHECK(cudaMemcpy(block->f_sending_colinds_->mutable_cpu_data(), block->f_sending_colinds_->gpu_data(), block->f_sending_colinds_->size(), cudaMemcpyDeviceToHost));
				for(unsigned int i = 0; i < block->f_sending_ranks_.size(); i++)
				{
					unsigned int shape = (block->f_sending_rowptrs_->cpu_data())[i + 1] - (block->f_sending_rowptrs_->cpu_data())[i];
					filename = string(rpath) + string("/") + string("sending_nid_") + to_string(rank) + string("_to_") + to_string(block->f_sending_ranks_[i]) + string(".npy");
					cnpy::npy_save<unsigned int>(filename, block->f_sending_colinds_->cpu_data() + (block->f_sending_rowptrs_->cpu_data())[i], {shape}, "w");
				}
			}
		}

		if(!block->f_receiving_ranks_.empty())
		{
			status.resize(block->f_receiving_ranks_.size());
			if(!check_cpu)
				CUDA_CHECK(cudaMemcpy(block->f_receiving_offsets_->mutable_cpu_data(), block->f_receiving_offsets_->gpu_data(), block->f_receiving_offsets_->size(), cudaMemcpyDeviceToHost));
			//unsigned int m = 0;
			for(unsigned int i = 0; i < block->f_receiving_ranks_.size(); i++)
			{
				unsigned int elems = (block->f_receiving_rowptrs_->cpu_data())[i + 1] - (block->f_receiving_rowptrs_->cpu_data())[i];
				#if 0
				assert(block->f_receiving_offsets_->cpu_data()[i] == m);
				m += (align_up<5, unsigned int>(elems) >> 3);
				if(rank == 0)
				{
					unsigned int j = block->f_receiving_ranks_[i];
					switch(j)
					{
						case 1:
							assert(elems == 402701);
							break;
						case 2:
							assert(elems == 402702);
							break;
						case 3:
							assert(elems == 398254);
							break;
						default:
							assert(0);
					}
				}
				else if(rank == 1)
				{
					switch(block->f_sending_ranks_[i])
					{
						case 0:
							assert(elems == 402701);
							break;
						case 2:
							assert(elems == 405144);
							break;
						case 3:
							assert(elems == 398254);
							break;
						default:
							assert(0);
					}
				}
				else if(rank == 2)
				{
					switch(block->f_sending_ranks_[i])
					{
						case 0:
							assert(elems == 402701);
							break;
						case 1:
							assert(elems == 398257);
							break;
						case 3:
							assert(elems == 402699);
							break;
						default:
							assert(0);
					}
				}
				else if(rank == 3)
				{
					switch(block->f_sending_ranks_[i])
					{
						case 0:
							assert(elems == 402705);
							break;
						case 1:
							assert(elems == 398256);
							break;
						case 2:
							assert(elems == 402698);
							break;
						default:
							assert(0);
					}
				}
				else
					assert(0);
				#endif
				elems = (align_up<3, unsigned int>(elems) >> 3);
				recv_bytes.push_back(elems);
			}

			if(check_cpu)
				receiving_buff_cpu.resize((block->f_receiving_rowptrs_->cpu_data())[block->f_receiving_ranks_.size()]);
			
			if(saving_exchange)
			{
				receiving_buff = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * (block->f_receiving_rowptrs_->cpu_data())[block->f_receiving_ranks_.size()]); 
				calibrated_recv_buff = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * (block->f_receiving_rowptrs_->cpu_data())[block->f_receiving_ranks_.size()]);

				for(unsigned int i = 0; i < block->f_receiving_ranks_.size(); i++)
				{
					unsigned int shape = (block->f_receiving_rowptrs_->cpu_data())[i + 1] - (block->f_receiving_rowptrs_->cpu_data())[i];
					filename = string(rpath) + string("/") + string("receiving_nid_") + to_string(rank) + string("_from_") + to_string(block->f_receiving_ranks_[i]) + string(".npy");
					cnpy::npy_save<unsigned int>(filename, block->f_receiving_colinds_->cpu_data() + (block->f_receiving_rowptrs_->cpu_data())[i], {shape}, "w");
				}
			}
		}

		tag++;
		MPI_Barrier(comm);
		//time_start = MPI_Wtime();
		
		for(int i = 0; i < iters; i++)
		{
			int sidx;
			block->update_time();
			block->update_V_membrane_gpu();
			
			if(saving_sample && i >= saving_number)
			{
				block->update_sample_neurons_gpu();
			}

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

			block->update_F_sending_actives_gpu(stream1->get());
			if(saving_exchange)
			{
				saving_sending_spike_gpu(block->get_F_actives_gpu(),
										block->f_sending_rowptrs_->gpu_data(),
										block->f_sending_ranks_.size(),
										block->f_sending_colinds_->gpu_data(),
										native_send_buff->mutable_gpu_data(),
										stream2->get());
				CUDA_CHECK(cudaMemcpyAsync(native_send_buff->mutable_cpu_data(), native_send_buff->gpu_data(), native_send_buff->size(), cudaMemcpyDeviceToHost, stream2->get()));
			}
			
			CUDA_CHECK(cudaStreamSynchronize(stream1->get()));
			
			if(!block->f_sending_ranks_.empty())
			{
				//CUDA_CHECK(cudaMemcpy(f_sending_actives->mutable_cpu_data(), block->f_sending_actives_->gpu_data(), f_sending_actives->size(), cudaMemcpyDeviceToHost));
				for(unsigned int j = 0; j < block->f_sending_ranks_.size(); j++)
				{
					assert(rank != block->f_sending_ranks_[j]);
					MPI_Isend(block->f_sending_actives_->gpu_data() + (block->f_sending_offsets_->cpu_data())[j], send_bytes[j], MPI_CHAR, block->f_sending_ranks_[j], tag, comm, &requests[j]);
					//MPI_Isend(f_sending_actives->cpu_data() + (block->f_sending_offsets_->cpu_data())[j], send_bytes[j], MPI_CHAR, block->f_sending_ranks_[j], tag, comm, &requests[j]);
				}

			}
			
			if(!block->f_receiving_ranks_.empty())
			{
				for(unsigned int j = 0; j < block->f_receiving_ranks_.size(); j++)
				{
					MPI_Recv(block->f_receiving_actives_->mutable_gpu_data() + (block->f_receiving_offsets_->cpu_data())[j], recv_bytes[j], MPI_CHAR, block->f_receiving_ranks_[j], tag, comm, &status[j]);
					//memset(f_receiving_actives->mutable_cpu_data(), 0x00, f_receiving_actives->size());
					//MPI_Recv(f_receiving_actives->mutable_cpu_data() + (block->f_receiving_offsets_->cpu_data())[j], recv_bytes[j], MPI_CHAR, block->f_receiving_ranks_[j], tag, comm, &status[j]);
					int elems;
					MPI_Get_count(&status[j], MPI_CHAR, &elems);
					assert(elems == recv_bytes[j]);
				}
				//CUDA_CHECK(cudaMemcpy(block->f_receiving_actives_->mutable_gpu_data(), f_receiving_actives->cpu_data(), f_receiving_actives->size(), cudaMemcpyHostToDevice));

				
				if(saving_exchange)
				{
					save_exchange_spike_gpu(block->f_receiving_actives_->gpu_data(),
										block->f_receiving_offsets_->gpu_data(),
										block->f_receiving_rowptrs_->gpu_data(),
										block->f_receiving_ranks_.size(),
										receiving_buff->mutable_gpu_data(),
										stream2->get());
					CUDA_CHECK(cudaMemcpyAsync(receiving_buff->mutable_cpu_data(), receiving_buff->gpu_data(), receiving_buff->size(), cudaMemcpyDeviceToHost, stream2->get()));
					CUDA_CHECK(cudaStreamSynchronize(stream2->get()));
					for(unsigned int j = 0; j < block->f_receiving_ranks_.size(); j++)
					{
						vector<size_t> shape(2);
						shape[0] = 1;
						shape[1] = (block->f_receiving_rowptrs_->cpu_data())[j + 1] - (block->f_receiving_rowptrs_->cpu_data())[j];
						filename = string(rpath) + string("/") + string("receiving_spike_") + to_string(rank) + string("_from_") + to_string(block->f_receiving_ranks_[j]) + string(".npy");
						if(i == 0)
							cnpy::npy_save<unsigned char>(filename, receiving_buff->cpu_data() + (block->f_receiving_rowptrs_->cpu_data())[j], shape, "w");
						else
							cnpy::npy_save<unsigned char>(filename, receiving_buff->cpu_data() + (block->f_receiving_rowptrs_->cpu_data())[j], shape, "a");
					}
				}
			}

			if(!block->f_sending_ranks_.empty())
			{
				/*
				for(unsigned int j = 0; j < block->f_sending_ranks_.size(); j++)
				{
					int err = MPI_Wait(&requests[j], MPI_STATUSES_IGNORE);
					assert(err == MPI_SUCCESS);
				}
				*/
				int err = MPI_Waitall(block->f_sending_ranks_.size(), requests.data(), MPI_STATUSES_IGNORE);
				assert(err == MPI_SUCCESS);

				if(saving_exchange)
				{
					save_exchange_spike_gpu(block->f_sending_actives_->gpu_data(),
										block->f_sending_offsets_->gpu_data(),
										block->f_sending_rowptrs_->gpu_data(),
										block->f_sending_ranks_.size(),
										sending_buff->mutable_gpu_data(),
										stream1->get());
					CUDA_CHECK(cudaMemcpyAsync(sending_buff->mutable_cpu_data(), sending_buff->gpu_data(), sending_buff->size(), cudaMemcpyDeviceToHost, stream1->get()));
					CUDA_CHECK(cudaStreamSynchronize(stream1->get()));
					for(unsigned int j = 0; j < block->f_sending_ranks_.size(); j++)
					{
						vector<size_t> shape(2);
						shape[0] = 1;
						shape[1] = (block->f_sending_rowptrs_->cpu_data())[j + 1] - (block->f_sending_rowptrs_->cpu_data())[j];
						filename = string(rpath) + string("/") + string("sending_spike_") + to_string(rank) + string("_to_") + to_string(block->f_sending_ranks_[j]) + string(".npy");
						if(i == 0)
							cnpy::npy_save<unsigned char>(filename, sending_buff->cpu_data() + (block->f_sending_rowptrs_->cpu_data())[j], shape, "w");
						else
							cnpy::npy_save<unsigned char>(filename, sending_buff->cpu_data() + (block->f_sending_rowptrs_->cpu_data())[j], shape, "a");
					}
				}
			}

			block->update_F_receiving_actives_gpu();
			block->update_J_presynaptic_inner_gpu();
			if(saving_exchange)
			{
				save_spike_gpu(block->get_outer_f_actives_gpu(),
							   calibrated_recv_buff->count(),
							   calibrated_recv_buff->mutable_gpu_data(),
							   stream2->get());
				CUDA_CHECK(cudaMemcpyAsync(calibrated_recv_buff->mutable_cpu_data(), calibrated_recv_buff->gpu_data(), calibrated_recv_buff->size(), cudaMemcpyDeviceToHost, stream2->get()));	
				CUDA_CHECK(cudaStreamSynchronize(stream2->get()));
			}
			
			
			block->update_J_presynaptic_outer_gpu();
			block->update_J_presynaptic_gpu();
			block->update_I_synaptic_gpu();

			if(saving_gpu)
			{
				save_spike_gpu(block->get_F_actives_gpu(),
							   total_neurons,
							   saving_buff->mutable_gpu_data(),
							   stream1->get());
				CUDA_CHECK(cudaMemcpyAsync(saving_buff->mutable_cpu_data(), saving_buff->gpu_data(), saving_buff->size(), cudaMemcpyDeviceToHost, stream1->get()));
				CUDA_CHECK(cudaStreamSynchronize(stream1->get()));
			}
			
			CUDA_CHECK(cudaDeviceSynchronize());
			
			
			if(saving_exchange)
			{
				for(unsigned int j = 0; j < sending_buff->count(); j++)
					assert(sending_buff->cpu_data()[j] == native_send_buff->cpu_data()[j]);
				
				for(unsigned int j = 0; j < receiving_buff->count(); j++)
					assert(receiving_buff->cpu_data()[j] == calibrated_recv_buff->cpu_data()[j]);
			}
			
			if(check_cpu)
			{
				block->update_V_membrane_cpu();
				if(preset_sample)
					block->update_F_active_from_preset_cpu(sample_vect[sidx]->cpu_data(), sample_vect[sidx]->count(), noise_rate);
				else
					block->update_F_active_cpu(noise_rate);
				
				block->update_F_sending_actives_cpu();
				if(!block->f_sending_ranks_.empty())
				{
					for(unsigned int j = 0; j < block->f_sending_ranks_.size(); j++)
					{
						assert(rank != block->f_sending_ranks_[j]);
						MPI_Isend(block->f_sending_actives_->cpu_data() + (block->f_sending_offsets_->cpu_data())[j], send_bytes[j], MPI_CHAR, block->f_sending_ranks_[j], tag, comm, &requests[j]);
					}

					if(saving_exchange)
					{
						save_exchange_spike_cpu(block->f_sending_actives_->cpu_data(),
										block->f_sending_offsets_->cpu_data(),
										block->f_sending_rowptrs_->cpu_data(),
										block->f_sending_ranks_.size(),
										sending_buff_cpu.data());
					}
				}
			
				if(!block->f_receiving_ranks_.empty())
				{
					for(unsigned int j = 0; j < block->f_receiving_ranks_.size(); j++)
					{
						MPI_Recv(block->f_receiving_actives_->mutable_cpu_data() + (block->f_receiving_offsets_->cpu_data())[j], recv_bytes[j], MPI_CHAR, block->f_receiving_ranks_[j], tag, comm, &status[j]);
						int elems;
						MPI_Get_count(&status[j], MPI_CHAR, &elems);
						assert(elems == recv_bytes[j]);
					}

					if(saving_exchange)
					{
						save_exchange_spike_cpu(block->f_receiving_actives_->cpu_data(),
										block->f_receiving_offsets_->cpu_data(),
										block->f_receiving_rowptrs_->cpu_data(),
										block->f_receiving_ranks_.size(),
										receiving_buff_cpu.data());
					}
				}

				block->update_F_receiving_actives_cpu();
				block->update_J_presynaptic_inner_cpu();
				block->update_J_presynaptic_outer_cpu();
				block->update_J_presynaptic_cpu();
				block->update_I_synaptic_cpu();

				
				(*os) << "========gpu vs. cpu====================" << endl;
				if(saving_exchange)
				{
					if(!block->f_sending_ranks_.empty())
					{
						assert(sending_buff->count() == sending_buff_cpu.size());
						unsigned int j = 0;
						for(; j < sending_buff->count(); j++)
						{	
							if(sending_buff->cpu_data()[j] != sending_buff_cpu[j])
							{
								(*os) << "the active flags of the " << i << "th sending exchange active is not equal" << endl;
								break;
							}
						}

						if( j == sending_buff->count())
							(*os) << "the active flags of the " << i << "th sending exchange active is same" << endl;
					}

					if(!block->f_receiving_ranks_.empty())
					{
						assert(receiving_buff->count() == receiving_buff_cpu.size());
						unsigned int j = 0;
						for(; j < receiving_buff->count(); j++)
						{	
							if(receiving_buff->cpu_data()[j] != receiving_buff_cpu[j])
							{
								(*os) << "the active flags of the " << i << "th receiving exchange active is not equal" << endl;
								break;
							}
						}

						if( j == receiving_buff->count())
							(*os) << "the active flags of the " << i << "th receiving exchange active is same" << endl;
					}
				}
				
				check_result<float, float2>(block, i, *os);
				
				if(!block->f_sending_ranks_.empty())
				{
					/*
					for(unsigned int j = 0; j < block->f_sending_ranks_.size(); j++)
					{
						int err = MPI_Wait(&requests[j], MPI_STATUSES_IGNORE);
						assert(err == MPI_SUCCESS);
					}
					*/
					int err = MPI_Waitall(block->f_sending_ranks_.size(), requests.data(), MPI_STATUSES_IGNORE);
					assert(err == MPI_SUCCESS);
				}
			}

			if(saving_sample && i >= saving_number)
			{
				if(nullptr != block->sample_indices_)
				{
					CUDA_CHECK(cudaMemcpy(block->f_sample_actives_->mutable_cpu_data(), block->f_sample_actives_->gpu_data(), block->f_sample_actives_->size(), cudaMemcpyDeviceToHost));
					CUDA_CHECK(cudaMemcpy(block->v_sample_membranes_->mutable_cpu_data(), block->v_sample_membranes_->gpu_data(), block->v_sample_membranes_->size(), cudaMemcpyDeviceToHost));
					vector<size_t> shape(2);
					shape[0] = 1;
					shape[1] = block->sample_indices_->count();
					filename = string(rpath) + string("/") + string("selection_spike_") + to_string(rank) + string(".npy");
					if(i == 0)
						cnpy::npy_save(filename, block->f_sample_actives_->cpu_data(), shape, "w");
					else
						cnpy::npy_save(filename, block->f_sample_actives_->cpu_data(), shape, "a");

					filename = string(rpath) + string("/") + string("selection_voltage_") + to_string(rank) + string(".npy");
					if(i == 0)
						cnpy::npy_save(filename, block->v_sample_membranes_->cpu_data(), shape, "w");
					else
						cnpy::npy_save(filename, block->v_sample_membranes_->cpu_data(), shape, "a");
				}
			}
			
			if(saving_gpu)
			{
				vector<size_t> shape(2);
				shape[0] = 1;
				shape[1] = 2 * block->get_total_neurons();
				if(i == 0)
				{
					filename = string(rpath) + string("/ex_presynaptic_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, reinterpret_cast<const float*>(block->get_J_ex_presynaptics_cpu()), shape, "w");

					filename = string(rpath) + string("/in_presynaptic_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, reinterpret_cast<const float*>(block->get_J_in_presynaptics_cpu()), shape, "w");

					shape[1] = block->get_total_neurons();

					filename = string(rpath) + string("/membrane_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, block->get_V_membranes_cpu(), shape, "w");

					filename = string(rpath) + string("/synaptic_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, block->get_I_synaptics_cpu(), shape, "w");

					filename = string(rpath) + string("/flag_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, saving_buff->cpu_data(), shape, "w");
				}
				else
				{
					filename = string(rpath) + string("/ex_presynaptic_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, reinterpret_cast<const float*>(block->get_J_ex_presynaptics_cpu()), shape, "a");

					filename = string(rpath) + string("/in_presynaptic_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, reinterpret_cast<const float*>(block->get_J_in_presynaptics_cpu()), shape, "a");

					shape[1] = block->get_total_neurons();

					filename = string(rpath) + string("/membrane_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, block->get_V_membranes_cpu(), shape, "a");

					filename = string(rpath) + string("/synaptic_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, block->get_I_synaptics_cpu(), shape, "a");

					filename = string(rpath) + string("/flag_") + to_string(rank) + string(".npy");
					cnpy::npy_save(filename, saving_buff->cpu_data(), shape, "a");
				}
			}
			
			tag++;
			MPI_Barrier(comm);
		}
	}
	
	MPI_Finalize();

	if(os != &cout) 
    	delete os;
	
	DEVICE_RESET
	
	return 0;
}

