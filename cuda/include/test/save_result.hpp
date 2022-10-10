#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace istbi {

void save_exchange_spike_gpu(const unsigned char* f_actives,
						const unsigned int* offsets,
						const unsigned int* rowptrs,
						const unsigned int n,
						unsigned char* f_saving_actives,
						cudaStream_t stream = NULL);

void save_exchange_spike_cpu(const unsigned char* f_actives,
						const unsigned int* offsets,
						const unsigned int* rowptrs,
						const unsigned int n,
						unsigned char* f_saving_actives);

void saving_sending_spike_gpu(const unsigned char* f_actives,
							const unsigned int* rowptrs,
							const unsigned int n,
							const unsigned int* colinds,
							unsigned char* f_saving_actives,
							cudaStream_t stream = NULL);

void save_spike_gpu(const unsigned char* f_actives,
						const unsigned int n,
						unsigned char* f_saving_actives,
						cudaStream_t stream = NULL);

template<typename T>
void resort_result_gpu(const long long* h_resort_inds,
						const T* d_keys,
						const unsigned int n,
						T* d_vals);

template<typename T>
void resort_params_cpu(unsigned long long* nids,
						unsigned int n,
						unsigned char* kinds,
						T* weights);

void resort_samples_gpu(long long* samples,
							const unsigned int height,
							const unsigned int width,
							const unsigned int bid,
							std::vector<unsigned int>& indices);

void resort_samples_cpu(long long* samples,
							const unsigned int height,
							const unsigned int width,
							const unsigned int bid,
							std::vector<unsigned int>& indices);



}//namespace istbi
