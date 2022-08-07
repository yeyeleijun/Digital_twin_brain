#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include "common.hpp"
#include "stage.hpp"

using namespace istbi;

int main(int argc, char **argv)
{
	CUDA_CHECK(cudaSetDevice(0));
	// Clear error status
  	CUDA_CHECK(cudaGetLastError());

	unsigned int h_rowptrs[6] = {0, 11, 31, 41, 50, 62};
	unsigned int* d_rowptrs;
	CUDA_CHECK(cudaMalloc((void**)&d_rowptrs, 6 * sizeof(unsigned int)));
	CUDA_CHECK(cudaMemcpy(d_rowptrs, h_rowptrs, 6 * sizeof(unsigned int), cudaMemcpyHostToDevice));
	unsigned char *h_actives, *d_actives;
	unsigned int bits = align_up<3, unsigned int>(11 - 0) + align_up<3, unsigned int>(31 - 11) +
						align_up<3, unsigned int>(41 - 31) + align_up<3, unsigned int>(50 - 41) + align_up<3, unsigned int>(62 - 50);
	unsigned int size = (align_up<2, unsigned int>(bits >> 3)) ;
	h_actives = (unsigned char*)malloc(size);
	for(unsigned int i = 0; i < 4; i++)
		h_actives[i] = 0xFF;
	
	for(unsigned int i = 4; i < size; i++)
		h_actives[i] = 0x00;
	CUDA_CHECK(cudaMalloc((void**)&d_actives, size));
	CUDA_CHECK(cudaMemcpy(d_actives, h_actives, size, cudaMemcpyHostToDevice));
	std::cout << "input size:" << size << std::endl;

	unsigned char *h_results, *d_results;
	size = (align_up<5, unsigned int>(64) >> 3);
	std::cout << "output size:" << size << std::endl;
	h_results = (unsigned char*)malloc(size);
	CUDA_CHECK(cudaMalloc((void**)&d_results, size));
	calibrate_receiving_spike_gpu(d_actives,
								d_rowptrs,
								5,
								64,
								d_results);
	
	CUDA_CHECK(cudaMemcpy(h_results, d_results, size, cudaMemcpyDeviceToHost));
	std::cout << "Result:" << std::endl;
	for(unsigned int i = 0; i < size; i++)
	{
		std::cout << static_cast<int>(h_results[i]);
		std::cout << " ";
	}
	std::cout << std::endl;

	cudaFree(d_rowptrs);
	cudaFree(d_actives);
	cudaFree(d_results);

	free(h_actives);
	free(h_results);
	return 0;
}

