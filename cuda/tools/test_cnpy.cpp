#include <string>
#include <iostream>
#include "util/cnpy.h"

using namespace std;

int main(int argc, char** argv)
{
	cnpy::NpyArray arr_nids = cnpy::npz_load(argv[1], "output_neuron_idx");
	unsigned int* nids = arr_nids.data<unsigned int>();
	assert(arr_nids.shape.size() == 1);

	if(!cnpy::npz_find(argv[1], "input_block_idx"))
		cerr << "cannot find variable name input_block_idx" << endl;

	cnpy::NpyArray arr_conn_bids = cnpy::npz_load(argv[1], "input_block_idx");
	unsigned short* conn_bids = arr_conn_bids.data<unsigned short>();
	assert(arr_conn_bids.shape.size() == 1);

	cnpy::NpyArray arr_conn_nids = cnpy::npz_load(argv[1], "input_neuron_idx");
	unsigned int* conn_nids = arr_conn_nids.data<unsigned int>();
	assert(arr_conn_nids.shape.size() == 1);

	return 0;
}

