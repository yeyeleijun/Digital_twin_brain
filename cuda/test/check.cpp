#include <iostream>
#include <string>
#include <float.h>
#include "common.hpp"
#include "test/check.hpp"
#include "util/transpose.hpp"
#include "util/cnpy.h"
#include "data_allocator.hpp"
#include "test/save_result.hpp"
#include <sys/types.h>
#include <dirent.h>

using namespace std;

namespace istbi {

#define SOFTENING 1e-9f

template<typename T, typename T2>
shared_ptr<BrainBlock<T, T2>> init_brain_block(const char* filename,
												const T delta_t,
												const unsigned int bid,
												const unsigned int gid)
{
	cnpy::npz_t arr = cnpy::npz_load(filename);

	cnpy::NpyArray arr_prop = arr["property"];
	T* props = arr_prop.data<T>();
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
	T2* weights = reinterpret_cast<T2*>(arr_weight.data<T>());
	assert(arr_weight.shape.size() == 2 &&
		arr_weight.shape[0] == arr_nids.shape[0] &&
		arr_weight.shape[1] == 2);
	
	shared_ptr<BrainBlock<T, T2>> shared_block = make_shared<BrainBlock<T, T2>>(arr_prop.shape[0], static_cast<unsigned short>(bid), gid, delta_t);
	BrainBlock<T, T2>* block = shared_block.get();
	block->init_connection_table_gpu(nids,
									conn_bids,
									conn_nids,
									conn_kinds,
									weights,
									arr_nids.shape[0]);
	block->init_config_params_gpu(props, arr_prop.shape[0], arr_prop.shape[1]);
	return shared_block;
}

template<typename T2>
static void check_params(const unsigned long long* conn_nids,
							const unsigned int* rowptrs,
							const unsigned int* colinds,
							const unsigned char* connkinds,
							const T2* weights,
							const unsigned int n,
							unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>>& connnid_maps,
							unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>>& connkind_maps,
							unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>>& connweight_maps)
{
	unordered_map<unsigned int, unsigned int> lastidx_maps;
	for(auto it = connnid_maps.begin(); it != connnid_maps.end(); ++it)
	{
		lastidx_maps.emplace(it->first, 0);
	}
	
	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int beg = rowptrs[i];
		unsigned int end = rowptrs[i + 1];
		for(unsigned int j = beg; j < end; j++)
		{
			const unsigned int nid = colinds[j];
			const unsigned int last_idx = lastidx_maps[nid];
			const unsigned long long conn_nid = (connnid_maps[nid])->cpu_data()[last_idx];
			const unsigned char conn_kind = (connkind_maps[nid])->cpu_data()[last_idx];
			const T2 conn_weight = (connweight_maps[nid])->cpu_data()[last_idx];
			
			assert(conn_nid == conn_nids[i]);
			assert(conn_kind == connkinds[j]);
			assert(conn_weight.x == weights[j].x &&
					conn_weight.y == weights[j].y);
			lastidx_maps[nid]++;
		}
	}
	
	for (auto it = lastidx_maps.begin(); it != lastidx_maps.end(); ++it)
	{
		assert(it->second == (connnid_maps[it->first])->count());
	}
}

template<typename T, typename T2>
static void read_params_from_numpy(const char* filename,
								unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>>& inner_conn_nid_maps,
								unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>>& inner_conn_kind_maps,
								unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>>& inner_conn_weight_maps,
								unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>>& outer_conn_nid_maps,
								unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>>& outer_conn_kind_maps,
								unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>>& outer_conn_weight_maps,
								unsigned int& nids)
{
	//load it into a new array
	cnpy::npz_t arr = cnpy::npz_load(filename);

	cnpy::NpyArray arr_idx = arr["idx"];
	unsigned int* indices = arr_idx.data<unsigned int>();
	assert(arr_idx.shape.size() == 2 && arr_idx.shape[0] == 4);
	unsigned int total = arr_idx.shape[1];

	cnpy::NpyArray arr_w = arr["weight"];
	T* weights = arr_w.data<T>();
	assert(arr_w.shape.size() == 1 && arr_w.shape[0] == arr_idx.shape[1]);

	string filestr(filename);
	string::size_type begin = filestr.rfind("_");
	string::size_type end = filestr.rfind(".");
	assert(begin != string::npos);
	char* str_end;
	unsigned int bid = static_cast<unsigned int>(strtoul(filestr.substr(begin + 1, end - begin - 1).c_str(), &str_end, 10));

	nids = 0;
	unsigned int idx = 0;
	while(idx < total)
	{			
		bool has_inner = false;
		bool has_outer = false;
		unsigned int inner_beg = 0;
		unsigned int inner_count = 0;
		unsigned int outer_beg = 0;
		unsigned int outer_count = 0;

		unsigned int nid = indices[idx];
		unsigned int i = idx;
		for(; i < total; i += 2)
		{
			assert(indices[total + i] == indices[total + i + 1] &&
				indices[2 * total + i] == indices[2 * total + i + 1]);
			if(nid == indices[i + 1])
			{
				if(bid == indices[total + i + 1])
				{
					if(!has_inner)
					{
						inner_beg = i;
						has_inner = true;
					}
					inner_count++;
				}
				else
				{
					if(!has_outer)
					{
						outer_beg = i;
						has_outer = true;
					}
					outer_count++;
				}
			}
			else
				break;
		}

		if(has_inner)
		{
			shared_ptr<DataAllocator<unsigned long long>> inner_connnids = make_shared<DataAllocator<unsigned long long>>(sizeof(unsigned long long) * inner_count);
			shared_ptr<DataAllocator<unsigned char>> inner_connkinds = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * inner_count);
			shared_ptr<DataAllocator<T2>> inner_connweights = make_shared<DataAllocator<T2>>(sizeof(T2) * inner_count);
			unsigned int count = 0;
			unsigned int end = inner_beg + (inner_count << 1);
			for(unsigned int j = inner_beg; j < end; j += 2, count++)
			{
				(inner_connnids->mutable_cpu_data())[count] = indices[2 * total + j];
				(inner_connkinds->mutable_cpu_data())[count] = static_cast<unsigned char>(indices[3 * total + j]);
				(inner_connweights->mutable_cpu_data())[count].x = weights[j];
				(inner_connweights->mutable_cpu_data())[count].y = weights[j + 1];
				
			}
			
			assert(count == inner_count);
			resort_params_cpu<T2>(inner_connnids->mutable_cpu_data(), inner_count, inner_connkinds->mutable_cpu_data(), inner_connweights->mutable_cpu_data());
			inner_conn_nid_maps.emplace(nid, inner_connnids);
			inner_conn_kind_maps.emplace(nid, inner_connkinds);
			inner_conn_weight_maps.emplace(nid, inner_connweights);
		}

		if(has_outer)
		{
			shared_ptr<DataAllocator<unsigned long long>> outer_connnids = make_shared<DataAllocator<unsigned long long>>(sizeof(unsigned long long) * outer_count);
			shared_ptr<DataAllocator<unsigned char>> outer_connkinds = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * outer_count);
			shared_ptr<DataAllocator<T2>> outer_connweights = make_shared<DataAllocator<T2>>(sizeof(T2) * outer_count);
			unsigned int count = 0;
			if(outer_beg < inner_beg)
			{
				for(unsigned int j = outer_beg; j < inner_beg; j += 2, count++)
				{
					const unsigned long long bid = indices[total + j];
					(outer_connnids->mutable_cpu_data())[count] = ((bid << 32) | indices[2 * total + j]);
					(outer_connkinds->mutable_cpu_data())[count] = static_cast<unsigned char>(indices[3 * total + j]);
					(outer_connweights->mutable_cpu_data())[count].x = weights[j];
					(outer_connweights->mutable_cpu_data())[count].y = weights[j + 1];
				}

				if(outer_count > count)
				{
					for(unsigned int j = inner_beg + (inner_count << 1); j < i; j += 2, count++)
					{
						const unsigned long long bid = indices[total + j];
						(outer_connnids->mutable_cpu_data())[count] = ((bid << 32) | indices[2 * total + j]);
						(outer_connkinds->mutable_cpu_data())[count] = static_cast<unsigned char>(indices[3 * total + j]);
						(outer_connweights->mutable_cpu_data())[count].x = weights[j];
						(outer_connweights->mutable_cpu_data())[count].y = weights[j + 1];
					}
				}
			}
			else
			{
				for(unsigned int j = outer_beg; j < i; j += 2, count++)
				{
					const unsigned long long bid = indices[total + j];
					(outer_connnids->mutable_cpu_data())[count] = ((bid << 32) | indices[2 * total + j]);
					(outer_connkinds->mutable_cpu_data())[count] = static_cast<unsigned char>(indices[3 * total + j]);
					(outer_connweights->mutable_cpu_data())[count].x = weights[j];
					(outer_connweights->mutable_cpu_data())[count].y = weights[j + 1];
				}
			}
			assert(count == outer_count);
			resort_params_cpu<T2>(outer_connnids->mutable_cpu_data(), outer_count, outer_connkinds->mutable_cpu_data(), outer_connweights->mutable_cpu_data());
			outer_conn_nid_maps.emplace(nid, outer_connnids);
			outer_conn_kind_maps.emplace(nid, outer_connkinds);
			outer_conn_weight_maps.emplace(nid, outer_connweights);
		}
		
		idx = i;
		nids++;
	}
	assert(idx == total);
}

template<typename T, typename T2>
void check_params(const char* spath, const char* mpath, const unsigned int blocks)
{
	unsigned int single_nids;
	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>> single_conn_nid_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>> single_conn_kind_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>> single_conn_weight_maps;

	unsigned int multi_nids = 0;
	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>> multi_conn_nid_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>> multi_conn_kind_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>> multi_conn_weight_maps;

	string filename = string(spath) + string("/block_0.npz");
	read_params_from_numpy<T, T2>(filename.c_str(),
						single_conn_nid_maps,
						single_conn_kind_maps,
						single_conn_weight_maps,
						multi_conn_nid_maps,
						multi_conn_kind_maps,
						multi_conn_weight_maps,
						single_nids);
	assert(multi_conn_nid_maps.empty());
	for(unsigned int i = 0; i < single_nids; i++)
	{
		resort_params_cpu<T2>(single_conn_nid_maps[i]->mutable_cpu_data(), single_conn_nid_maps[i]->count(), single_conn_kind_maps[i]->mutable_cpu_data(), single_conn_weight_maps[i]->mutable_cpu_data());
	}

	filename = string(spath) + string("/resort_idx.npy");
	cnpy::NpyArray arr = cnpy::npy_load(filename);
	long long* resort_inds = arr.data<long long>();
	assert(arr.shape.size() == 1 && arr.shape[0] == single_nids);
	vector<unsigned int> nids_vect;
	
	for(unsigned int i = 0; i < blocks; i++)
	{
		unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>> inner_conn_nid_maps;
		unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>> inner_conn_kind_maps;
		unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>> inner_conn_weight_maps;
	
		unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>> outer_conn_nid_maps;
		unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>> outer_conn_kind_maps;
		unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>> outer_conn_weight_maps;
		
		filename = string(mpath) + string("/block_") + to_string(i) + string(".npz");
		unsigned int nids;
		read_params_from_numpy<T, T2>(filename.c_str(),
									inner_conn_nid_maps,
									inner_conn_kind_maps,
									inner_conn_weight_maps,
									outer_conn_nid_maps,
									outer_conn_kind_maps,
									outer_conn_weight_maps,
									nids);

		for(unsigned int j = 0; j < nids; j++)
		{
			unsigned int inner_count = 0;
			unsigned int outer_count = 0;
			if(inner_conn_nid_maps.find(j) != inner_conn_nid_maps.end())
			{
				inner_count = inner_conn_nid_maps[j]->count();
			}

			if(outer_conn_nid_maps.find(j) != outer_conn_nid_maps.end())
			{
				outer_count = outer_conn_nid_maps[j]->count();
			}

			unsigned int count = inner_count + outer_count;
			if(count)
			{
				unsigned int nid = static_cast<unsigned int>(resort_inds[multi_nids + j]);
				
				shared_ptr<DataAllocator<unsigned long long>> connnids = make_shared<DataAllocator<unsigned long long>>(sizeof(unsigned long long) * count);
				shared_ptr<DataAllocator<unsigned char>> connkinds = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * count);
				shared_ptr<DataAllocator<T2>> connweights = make_shared<DataAllocator<T2>>(sizeof(T2) * count);
				if(inner_count)
				{
					for(unsigned int k = 0; k < inner_count; k++)
					{
						unsigned long long bid = i;
						connnids->mutable_cpu_data()[k] = ((bid << 32) | inner_conn_nid_maps[j]->cpu_data()[k]);
					}
					memcpy(connkinds->mutable_cpu_data(), inner_conn_kind_maps[j]->cpu_data(), inner_count * sizeof(unsigned char));
					memcpy(connweights->mutable_cpu_data(), inner_conn_weight_maps[j]->cpu_data(), inner_count * sizeof(T2));
				}
				
				if(outer_count)
				{
					memcpy(connnids->mutable_cpu_data() + inner_count, outer_conn_nid_maps[j]->cpu_data(), outer_count * sizeof(unsigned long long));
					memcpy(connkinds->mutable_cpu_data() + inner_count, outer_conn_kind_maps[j]->cpu_data(), outer_count * sizeof(unsigned char));
					memcpy(connweights->mutable_cpu_data() + inner_count, outer_conn_weight_maps[j]->cpu_data(), outer_count * sizeof(T2));
				}

				multi_conn_nid_maps.emplace(nid, connnids);
				multi_conn_kind_maps.emplace(nid, connkinds);
				multi_conn_weight_maps.emplace(nid, connweights);
			}
		}

		nids_vect.push_back(multi_nids);
		multi_nids += nids;
	}

	assert(multi_nids == single_nids);
	assert(!nids_vect.empty());

	for(unsigned int i = 0; i < multi_nids; i++)
	{
		assert(multi_conn_nid_maps.find(i) != multi_conn_nid_maps.end());
		for(unsigned int j = 0; j < multi_conn_nid_maps[i]->count(); j++)
		{
			unsigned long long conn_nid = multi_conn_nid_maps[i]->cpu_data()[j];
			unsigned int conn_bid = static_cast<unsigned int>(conn_nid >> 32);
			multi_conn_nid_maps[i]->mutable_cpu_data()[j] = resort_inds[nids_vect[conn_bid] + static_cast<unsigned int>(conn_nid & 0xFFFFFFFFu)];
		}
		resort_params_cpu<T2>(multi_conn_nid_maps[i]->mutable_cpu_data(), multi_conn_nid_maps[i]->count(), multi_conn_kind_maps[i]->mutable_cpu_data(), multi_conn_weight_maps[i]->mutable_cpu_data());
	}

	for(unsigned int i = 0; i < multi_nids; i++)
	{
		assert(single_conn_nid_maps.find(i) != single_conn_nid_maps.end() &&
			multi_conn_nid_maps.find(i) != multi_conn_nid_maps.end() &&
			single_conn_nid_maps[i]->count() == multi_conn_nid_maps[i]->count());

		for(unsigned int j = 0; j < single_conn_nid_maps[i]->count(); j++)
		{
			assert(single_conn_nid_maps[i]->cpu_data()[j] == multi_conn_nid_maps[i]->cpu_data()[j]);
			assert(single_conn_kind_maps[i]->cpu_data()[j] == multi_conn_kind_maps[i]->cpu_data()[j]);
			assert(single_conn_weight_maps[i]->cpu_data()[j].x == multi_conn_weight_maps[i]->cpu_data()[j].x &&
				single_conn_weight_maps[i]->cpu_data()[j].y == multi_conn_weight_maps[i]->cpu_data()[j].y);
		}
	}
	
}

template<typename T, typename T2>
void check_params(BrainBlock<T, T2>* block, const char* filename)
{
	unsigned int nids;
	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>> inner_conn_nid_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>> inner_conn_kind_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>> inner_conn_weight_maps;

	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned long long>>> outer_conn_nid_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<unsigned char>>> outer_conn_kind_maps;
	unordered_map<unsigned int, shared_ptr<DataAllocator<T2>>> outer_conn_weight_maps;

	read_params_from_numpy<T, T2>(filename,
						inner_conn_nid_maps,
						inner_conn_kind_maps,
						inner_conn_weight_maps,
						outer_conn_nid_maps,
						outer_conn_kind_maps,
						outer_conn_weight_maps,
						nids);
	if(nids != block->get_total_neurons())
		cout << "============" << nids << " vs. " << block->get_total_neurons() << "===========" << endl;
	assert(nids == block->get_total_neurons());

	if(!inner_conn_nid_maps.empty())
	{
		unsigned int n = block->inner_conn_size();
		vector<unsigned long long> conn_nids(n);
		for(unsigned int i = 0; i < n; i++)
		{
			conn_nids[i] = (block->get_f_inner_conninds_cpu())[i];
		}
		check_params<T2>(conn_nids.data(),
						block->get_inner_rowptrs_cpu(),
						block->get_inner_colinds_cpu(),
						block->get_inner_connkinds_cpu(),
						block->get_inner_w_synaptics_cpu(),
						block->inner_conn_size(),
						inner_conn_nid_maps,
						inner_conn_kind_maps,
						inner_conn_weight_maps);
	}
	
	if(!outer_conn_nid_maps.empty())
	{
		unsigned int n = block->f_receiving_colinds_->count();
		vector<unsigned long long> conn_nids(n);
		for(unsigned int i = 0; i < block->f_receiving_ranks_.size(); i++)
		{
			unsigned long long bid = block->f_receiving_ranks_[i];
			for(unsigned int j = block->f_receiving_rowptrs_->cpu_data()[i]; j < block->f_receiving_rowptrs_->cpu_data()[i + 1]; j++)
			{
				unsigned int nid = block->f_receiving_colinds_->cpu_data()[j];
				conn_nids[j] = ((bid << 32) | nid);
			}
		}
		
		check_params<T2>(conn_nids.data(),
						block->get_outer_rowptrs_cpu(),
						block->get_outer_colinds_cpu(),
						block->get_outer_connkinds_cpu(),
						block->get_outer_w_synaptics_cpu(),
						block->f_receiving_colinds_->count(),
						outer_conn_nid_maps,
						outer_conn_kind_maps,
						outer_conn_weight_maps);
	}
}

template<typename T, typename T2>
static void check_result(const T2* src_j_ex_presynaptics,
						   const T2* src_j_in_presynaptics,
						   const T* src_v_membranes,
						   const T* src_i_synaptics,
						   const unsigned char* src_f_actives,
						   const unsigned int iter,
						   const unsigned int n,
						   const T2* dst_j_ex_presynaptics,
						   const T2* dst_j_in_presynaptics,
						   const T* dst_v_membranes,
						   const T* dst_i_synaptics,
						   const unsigned char* dst_f_actives,
						   ostream& out)
{
	float max_err;
	if(	NULL != src_v_membranes &&
		NULL != dst_v_membranes)
	{
		unsigned int max_idx = -1;
		T max_val;
		max_err = FLT_MIN;
		
		for(unsigned int i = 0; i < n; i++)
		{
			float diff = fabsf(dst_v_membranes[i] - src_v_membranes[i]) / fabsf(src_v_membranes[i] + SOFTENING);
			if(diff > max_err)
			{
				max_err = diff;
				max_idx = i;
				max_val = src_v_membranes[i];
			}
		}
		
		if(max_err > FLT_MIN)
		{
			out << "the max relative error of the " << iter << "th V membrane: " << max_err << endl;
			out << "gpu result: " << max_val << endl;
			out << "cpu result: " << dst_v_membranes[max_idx] << endl;
		}
		else
		{
			out << "the max relative error of the " << iter << "th V membrane: 0" << endl;
		}
	}

	{
		max_err = FLT_MIN;
		unsigned int num;
		unsigned int max_idx = -1;
		T max_val;
		bool has_ex_j = false;
		bool has_in_j = false;
		
		if(NULL != src_j_ex_presynaptics &&
			NULL != dst_j_ex_presynaptics)
		{
			has_ex_j = true;
			for(unsigned int i = 0; i < n; i++)
			{
				float diff = fabsf(dst_j_ex_presynaptics[i].x - src_j_ex_presynaptics[i].x) / fabsf(src_j_ex_presynaptics[i].x + SOFTENING);
				if(diff > max_err)
				{
					max_err = diff;
					max_idx = i;
					max_val = src_j_ex_presynaptics[i].x;
					num = 0;
				}
			
				diff = fabsf(dst_j_ex_presynaptics[i].y - src_j_ex_presynaptics[i].y) / fabsf(src_j_ex_presynaptics[i].y + SOFTENING);
				if(diff > max_err)
				{
					max_err = diff;
					max_idx = i;
					max_val = src_j_ex_presynaptics[i].y;
					num = 1;
				}
			}
			
		}
	
		if(NULL != dst_j_in_presynaptics &&
			NULL != src_j_in_presynaptics)
		{
			has_in_j= true;
			for(unsigned int i = 0; i < n; i++)
			{
				float diff = fabsf(dst_j_in_presynaptics[i].x - src_j_in_presynaptics[i].x) / fabsf(src_j_in_presynaptics[i].x + SOFTENING);
				if(diff > max_err)
				{
					max_err = diff;
					max_idx = i;
					max_val = src_j_in_presynaptics[i].x;
					num = 2;
				}
				
				diff = fabsf(dst_j_in_presynaptics[i].y - src_j_in_presynaptics[i].y) / fabsf(src_j_in_presynaptics[i].y + SOFTENING);
				if(diff > max_err)
				{
					max_err = diff;
					max_idx = i;
					max_val = src_j_in_presynaptics[i].y;
					num = 3;
				}
			}
		}

		if(has_ex_j || has_in_j)
		{
			if(max_err > FLT_MIN)
			{
				if(has_ex_j && !has_in_j)
					out << "the max relative error of the " << iter << "th excit J presynaptic: " << max_err << " at " << max_idx << "th neuron" << endl;
				else if(has_in_j && !has_ex_j)
					out << "the max relative error of the " << iter << "th inhbit J presynaptic: " << max_err << " at " << max_idx << "th neuron" << endl;
				else
					out << "the max relative error of the " << iter << "th J presynaptic: " << max_err << " at " << max_idx << "th neuron" << endl;
				//out.precision(8);
				out << "gpu result: " << max_val << endl;
				switch(num)
				{
					case 0:
						out << "cpu result: " << dst_j_ex_presynaptics[max_idx].x << endl;
						break;
					case 1:
						out << "cpu result: " << dst_j_ex_presynaptics[max_idx].y << endl;
						break;
					case 2:
						out << "cpu result: " << dst_j_in_presynaptics[max_idx].x << endl;
						break;
					default:
						out << "cpu result: " << dst_j_in_presynaptics[max_idx].y << endl;
						break;
				}
			}
			else
			{
				if(has_ex_j && !has_in_j)
					out << "the max relative error of the " << iter << "th J excit presynaptic: 0" << endl;
				else if(has_in_j && !has_ex_j)
					out << "the max relative error of the " << iter << "th J inhbit presynaptic: 0" << endl;
				else
					out << "the max relative error of the " << iter << "th J presynaptic: 0" << endl;
			}
		}
	}
	
	if(NULL != src_i_synaptics &&
		NULL != dst_i_synaptics)
	{
		max_err = FLT_MIN;
		unsigned int max_idx = -1;
		T max_val;

		for(unsigned int i = 0; i < n; i++)
		{
			float diff = fabsf(dst_i_synaptics[i] - src_i_synaptics[i]) / fabsf(src_i_synaptics[i] + SOFTENING);
			if(diff > max_err)
			{
				max_err = diff;
				max_idx = i;
				max_val = src_i_synaptics[i];
			}
		}
		
		if(max_err > FLT_MIN)
		{
			out << "the max relative error of the " << iter << "th I synaptic: " << max_err << " at "<< max_idx << "th neuron" << endl;
			//cout.precision(8);
			out << "gpu result: ";
			if(NULL != src_v_membranes)
				out << src_v_membranes[max_idx] << "\t";
			if(NULL != src_j_ex_presynaptics)
				out << src_j_ex_presynaptics[max_idx].x << "\t" << src_j_ex_presynaptics[max_idx].y << "\t";
			if(NULL != src_j_in_presynaptics)
				out << src_j_in_presynaptics[max_idx].x << "\t" << src_j_in_presynaptics[max_idx].y << "\t";
			out << max_val << endl;

			out << "cpu result: ";
			if(NULL != dst_v_membranes)
				out << dst_v_membranes[max_idx] << "\t";
			if(NULL != dst_j_ex_presynaptics)
				out << dst_j_ex_presynaptics[max_idx].x << "\t" << dst_j_ex_presynaptics[max_idx].y << "\t";
			if(NULL != dst_j_in_presynaptics)
				out << dst_j_in_presynaptics[max_idx].x << "\t" << dst_j_in_presynaptics[max_idx].y << "\t";
			out << dst_i_synaptics[max_idx] << endl;
		}
		else
		{
			out << "the max relative error of the " << iter << "th I synaptic: 0" << endl;
		}
	}

	if(NULL != src_f_actives &&
		NULL != dst_f_actives)
	{
		unsigned int i = 0;
		for(; i < n; i++)
		{	
			if(dst_f_actives[i] != src_f_actives[i])
			{
				out << "the active number of the " << iter << "th F active is not equal" << endl;
				break;
			}
		}

		if( i == n)
			out << "the active number of the " << iter << "th F active is same" << endl;
		
	}
}


template<typename T, typename T2>
static void check_result(BrainBlock<T, T2>* block,
						   const unsigned int iter,
						   const unsigned int n,
						   const T2* j_ex_presynaptics,
						   const T2* j_in_presynaptics,
						   const T* v_membranes,
						   const T* i_synaptics,
						   const unsigned char* f_actives,
						   ostream& out)
{
	assert(block->get_total_neurons() == n);
	vector<T2> presyn_ex(block->get_total_neurons());
	CUDA_CHECK(cudaMemcpy(presyn_ex.data(), block->get_J_ex_presynaptics_gpu(), block->get_total_neurons() * sizeof(T2), cudaMemcpyDeviceToHost));

	vector<T2> presyn_in(block->get_total_neurons());
	CUDA_CHECK(cudaMemcpy(presyn_in.data(), block->get_J_in_presynaptics_gpu(), block->get_total_neurons() * sizeof(T2), cudaMemcpyDeviceToHost));
	
	vector<T> memb(block->get_total_neurons());
	CUDA_CHECK(cudaMemcpy(memb.data(), block->get_V_membranes_gpu(), block->get_total_neurons() * sizeof(T), cudaMemcpyDeviceToHost));

	vector<T> syn(block->get_total_neurons());
	CUDA_CHECK(cudaMemcpy(syn.data(), block->get_I_synaptics_gpu(), block->get_total_neurons() * sizeof(T), cudaMemcpyDeviceToHost));

	shared_ptr<DataAllocator<unsigned char>> flags = make_shared<DataAllocator<unsigned char>>(sizeof(unsigned char) * block->get_total_neurons());
	save_spike_gpu(block->get_F_actives_gpu(), block->get_total_neurons(), flags->mutable_gpu_data());
	CUDA_CHECK(cudaMemcpy(flags->mutable_cpu_data(), flags->gpu_data(), flags->size(), cudaMemcpyDeviceToHost));
	
	check_result<T, T2>(presyn_ex.data(), presyn_in.data(),
						memb.data(), syn.data(), flags->cpu_data(),
						iter, block->get_total_neurons(),
						j_ex_presynaptics, j_in_presynaptics,
						v_membranes, i_synaptics, f_actives, out);
}



template<typename T, typename T2>
void check_result(BrainBlock<T, T2>* block, const char* filename, const unsigned int iter, ostream& out)
{
	cnpy::npz_t arr = cnpy::npz_load(filename);

	cnpy::NpyArray arr_memb = arr["membrane"];
	T* v_membranes = arr_memb.data<T>();
	assert(arr_memb.shape.size() == 1 && arr_memb.shape[0] == block->get_total_neurons());

	cnpy::NpyArray arr_presyn = arr["presynaptic"];
	T* presyn = arr_presyn.data<T>();
	assert(arr_presyn.shape.size() == 2 && arr_presyn.shape[0] == 4 && arr_presyn.shape[1] == block->get_total_neurons());
	vector<T2> j_ex_presynaptics;
	j_ex_presynaptics.resize(block->get_total_neurons());
	vector<T2> j_in_presynaptics;
	j_in_presynaptics.resize(block->get_total_neurons());
	for(unsigned int i = 0; i < block->get_total_neurons(); i++)
	{
		j_ex_presynaptics[i].x = presyn[i];
		j_ex_presynaptics[i].y = presyn[block->get_total_neurons() + i];
		j_in_presynaptics[i].x = presyn[2 * block->get_total_neurons() + i];
		j_in_presynaptics[i].y = presyn[3 * block->get_total_neurons() + i];
	}

	cnpy::NpyArray arr_syn = arr["synaptic"];
	T* i_synaptics = arr_syn.data<T>();
	assert(arr_syn.shape.size() == 1 && arr_syn.shape[0] == block->get_total_neurons());

	cnpy::NpyArray arr_flag = arr["flag"];
	unsigned char* f_actives = arr_flag.data<unsigned char>();
	assert(arr_flag.shape.size() == 1 && arr_flag.shape[0] == block->get_total_neurons());

	check_result<T, T2>(block,
					 iter,
					 block->get_total_neurons(),
					 j_ex_presynaptics.data(),
					 j_in_presynaptics.data(),
					 v_membranes,
					 i_synaptics,
					 f_actives,
					 out);
}

template<typename T, typename T2>
void check_result(BrainBlock<T, T2>* block, const unsigned int iter, ostream& out)
{
	const unsigned char* fp = block->get_F_actives_cpu(false);
	vector<unsigned char> flag(block->get_total_neurons());
	for(unsigned int i = 0; i < block->get_total_neurons(); i++)
	{
		unsigned char f = fp[i >> 3];
		flag[i] = ((f >> (i & 7)) & 0x1);
	}
	
	check_result<T, T2>(block, iter, block->get_total_neurons(), 
						block->get_J_ex_presynaptics_cpu(false), block->get_J_in_presynaptics_cpu(false),
						block->get_V_membranes_cpu(false), block->get_I_synaptics_cpu(false), flag.data(), out);
}

template<typename T, typename T2>
void check_result(const char* mb_filepath, const unsigned int blks, const char* sb_filepath, ostream& out)
{	
	unsigned int iter;
	unsigned long long n;
	unsigned long long m;
	string filename;

	{
		filename = string(sb_filepath) + string("/membrane_0.npy");
		cnpy::NpyArray sb_memb = cnpy::npy_load(filename);
		T* v_membranes = sb_memb.data<T>();
		assert(sb_memb.shape.size() == 2);
		iter = sb_memb.shape[0];
		n = sb_memb.shape[1];

		vector<T> v_membs(iter * n);
		unsigned int offset = 0;
		for(unsigned int i = 0; i < blks; i++)
		{
			filename = string(mb_filepath) + string("/membrane_") + to_string(i) + string(".npy");
			cnpy::NpyArray mb_memb = cnpy::npy_load(filename);
			T* v_memb = mb_memb.data<T>();
			assert(mb_memb.shape.size() == 2 &&
				mb_memb.shape[0] == iter);
			
			m = mb_memb.shape[1];
			for(unsigned int j = 0; j < iter; j++)
			{
				memcpy(v_membs.data() + j * n + offset, v_memb + j * m, sizeof(T) * m);
			}
			offset += m;
		}

		assert(offset == n);
		for(unsigned int j = 0; j < iter; j++)
		{
			check_result<T, T2>(NULL,
							NULL,
							v_membs.data() + j * n,
							NULL,
							NULL,
							j,
							static_cast<unsigned int>(n),
							NULL,
							NULL,
							v_membranes + j * n,
							NULL,
							NULL,
							out);	
		}
	}

	{
		filename = string(sb_filepath) + string("/ex_presynaptic_0.npy");
		cnpy::NpyArray sb_expresyn = cnpy::npy_load(filename);
		T2* j_ex_presynaptics = reinterpret_cast<T2*>(sb_expresyn.data<T>());
		assert(sb_expresyn.shape.size() == 2 && 
			sb_expresyn.shape[0] == iter &&
			sb_expresyn.shape[1] == (2 * n));

		vector<T2> j_ex_presyns(iter * n);
		unsigned int offset = 0;
		for(unsigned int i = 0; i < blks; i++)
		{
			filename = string(mb_filepath) + string("/ex_presynaptic_") + to_string(i) + string(".npy");
			cnpy::NpyArray mb_expresyn = cnpy::npy_load(filename);
			T2* j_ex_presyn = reinterpret_cast<T2*>(mb_expresyn.data<T>());
			assert(mb_expresyn.shape.size() == 2 && 
				mb_expresyn.shape[0] == iter);
			
			m = mb_expresyn.shape[1] >> 1;
			for(unsigned int j = 0; j < iter; j++)
			{
				memcpy(j_ex_presyns.data() + j * n + offset, j_ex_presyn + j * m, sizeof(T2) * m);
			}
			offset += m;
		}

		assert(offset == n);
		for(unsigned int j = 0; j < iter; j++)
		{
			check_result<T, T2>(j_ex_presyns.data() + j * n,
							NULL,
							NULL,
							NULL,
							NULL,
							j,
							static_cast<unsigned int>(n),
							j_ex_presynaptics + j * n,
							NULL,
							NULL,
							NULL,
							NULL,
							out);	
		}
	}

	{
		filename = string(sb_filepath) + string("/in_presynaptic_0.npy");
		cnpy::NpyArray sb_inpresyn = cnpy::npy_load(filename);
		T2* j_in_presynaptics = reinterpret_cast<T2*>(sb_inpresyn.data<T>());
		assert(sb_inpresyn.shape.size() == 2 &&
			sb_inpresyn.shape[0] == iter &&
			sb_inpresyn.shape[1] == (2 * n));

		vector<T2> j_in_presyns(iter * n);
		unsigned int offset = 0;
		for(unsigned int i = 0; i < blks; i++)
		{
			filename = string(mb_filepath) + string("/in_presynaptic_") + to_string(i) + string(".npy");
			cnpy::NpyArray mb_inpresyn = cnpy::npy_load(filename);
			T2* j_in_presyn = reinterpret_cast<T2*>(mb_inpresyn.data<T>());
			assert(mb_inpresyn.shape.size() == 2 &&
				mb_inpresyn.shape[0] == iter);
			
			m = mb_inpresyn.shape[1] >> 1;
			for(unsigned int j = 0; j < iter; j++)
			{
				memcpy(j_in_presyns.data() + j * n + offset, j_in_presyn + j * m, sizeof(T2) * m);
			}
			offset += m;
		}

		assert(offset == n);
		for(unsigned int j = 0; j < iter; j++)
		{
			check_result<T, T2>(NULL,
							j_in_presyns.data() + j * n,
							NULL,
							NULL,
							NULL,
							j,
							static_cast<unsigned int>(n),
							NULL,
							j_in_presynaptics + j * n,
							NULL,
							NULL,
							NULL,
							out);	
		}
	}

	{
		filename = string(sb_filepath) + string("/synaptic_0.npy");
		cnpy::NpyArray sb_synaptics = cnpy::npy_load(filename);
		T* i_synaptics = sb_synaptics.data<T>();
		assert(sb_synaptics.shape.size() == 2 &&
			sb_synaptics.shape[0] == iter &&
			sb_synaptics.shape[1] == n);

		vector<T> i_syns(iter * n);
		unsigned int offset = 0;
		for(unsigned int i = 0; i < blks; i++)
		{
			filename = string(mb_filepath) + string("/synaptic_") + to_string(i) + string(".npy");
			cnpy::NpyArray mb_synaptics = cnpy::npy_load(filename);
			T* i_syn = mb_synaptics.data<T>();
			assert(mb_synaptics.shape.size() == 2 &&
				mb_synaptics.shape[0] == iter);

			m = mb_synaptics.shape[1];
			for(unsigned int j = 0; j < iter; j++)
			{
				memcpy(i_syns.data() + j * n + offset, i_syn + j * m, sizeof(T) * m);
			}
			offset += m;
		}

		assert(offset == n);
		for(unsigned int j = 0; j < iter; j++)
		{
			check_result<T, T2>(NULL,
								NULL,
								NULL,
								i_syns.data() + j * n,
								NULL,
								j,
								static_cast<unsigned int>(n),
								NULL,
								NULL,
								NULL,
								i_synaptics + j * n,
								NULL,
								out);
		}
	}

	{
		filename = string(sb_filepath) + string("/flag_0.npy");
		cnpy::NpyArray sb_flags = cnpy::npy_load(filename);
		unsigned char* f_actives = sb_flags.data<unsigned char>();
		assert(sb_flags.shape.size() == 2 &&
			sb_flags.shape[0] == iter &&
			sb_flags.shape[1] == n);

		vector<unsigned char> f_acts(iter * n);
		unsigned int offset = 0;
		for(unsigned int i = 0; i < blks; i++)
		{
			filename = string(mb_filepath) + string("/flag_") + to_string(i) + string(".npy");
			cnpy::NpyArray mb_flags = cnpy::npy_load(filename);
			unsigned char* f_act = mb_flags.data<unsigned char>();
			assert(mb_flags.shape.size() == 2 &&
				mb_flags.shape[0] == iter);

			m = mb_flags.shape[1];
			for(unsigned int j = 0; j < iter; j++)
			{
				memcpy(f_acts.data() + j * n + offset, f_act + j * m, sizeof(unsigned char) * m);
			}

			offset += m;
		}
		assert(offset == n);
		for(unsigned int j = 0; j < iter; j++)
		{
			check_result<T, T2>(NULL,
								NULL,
								NULL,
								NULL,
								f_acts.data() + j * n,
								j,
								static_cast<unsigned int>(n),
								NULL,
								NULL,
								NULL,
								NULL,
								f_actives + j * n,
								out);
		}
	}
}

static void search_files_from_dir(const char* path,
									const char* prefix, 
									const char* comp_prefix,
									vector<string>& flist)
{
	DIR* pdir;
	struct dirent *pent;
	vector<string> comp_flist;
	assert((pdir = opendir(path)) != NULL);
	while((pent = readdir(pdir)) != NULL)
	{
		if (!strncasecmp(pent->d_name, prefix, strlen(prefix)))
		{
			flist.push_back(string(pent->d_name));
		}
		else if(!strncasecmp(pent->d_name, comp_prefix, strlen(comp_prefix)))
		{
			comp_flist.push_back(string(pent->d_name));
		}
	}

	assert(flist.size() == comp_flist.size());
	closedir(pdir);
}

void check_exchange_spike(const char* path, ostream& out)
{
	static const char* prefix = "sending_spike";
	static const char* comp_prefix = "receiving_spike";
	
	vector<string> flist;
	search_files_from_dir(path, prefix, comp_prefix, flist);
	
	for(auto it = flist.begin(); it != flist.end(); it++)
	{
		string::size_type begin = (*it).find("_", strlen(prefix));
		string::size_type end = (*it).find("_", begin + 1);
		assert(begin != string::npos);
		char* str_end;
		unsigned int src_rank = static_cast<unsigned int>(strtoul((*it).substr(begin + 1, end - begin - 1).c_str(), &str_end, 10));
		begin = (*it).find("_", end + 1);
		end = (*it).find(".", begin + 1);
		unsigned int dst_rank = static_cast<unsigned int>(strtoul((*it).substr(begin + 1, end - begin - 1).c_str(), &str_end, 10));
		string src_filename = string(path) + string("/") + string(prefix) + string("_") + to_string(src_rank) + string("_to_") + to_string(dst_rank) + string(".npy");
		string dst_filename = string(path) + string("/") + string(comp_prefix) + string("_") + to_string(dst_rank) + string("_from_") + to_string(src_rank) + string(".npy");

		//load it into a new array
		cnpy::NpyArray src_arr = cnpy::npy_load(src_filename);
		unsigned char* sending_spikes = src_arr.data<unsigned char>();
		assert(src_arr.word_size == sizeof(unsigned char));
		assert(src_arr.shape.size() == 2);

		cnpy::NpyArray dst_arr = cnpy::npy_load(dst_filename);
		unsigned char* receiving_spikes = dst_arr.data<unsigned char>();
		assert(dst_arr.word_size == sizeof(unsigned char));
		assert(dst_arr.shape.size() == 2);

		assert(src_arr.shape == dst_arr.shape);
		for(unsigned int i = 0; i < src_arr.shape[0]; i++)
		{
			unsigned int j = 0;
			for(; j < src_arr.shape[1]; j++)
			{
				if(sending_spikes[i * src_arr.shape[1] + j] != receiving_spikes[i * dst_arr.shape[1] + j])
				{
					out << "the " << i << "th exchange spike is not equal" << endl;
					break;
				}
			}
			
			if( j == src_arr.shape[1])
				out << "the " << i << "th exchange spike is same" << endl;
		}
	}
}

void check_exchange_nid(const char* path, ostream& out)
{
	static const char* prefix = "sending_nid";
	static const char* comp_prefix = "receiving_nid";
	
	vector<string> flist;
	search_files_from_dir(path, prefix, comp_prefix, flist);

	for(auto it = flist.begin(); it != flist.end(); it++)
	{
		string::size_type begin = (*it).find("_", strlen(prefix));
		string::size_type end = (*it).find("_", begin + 1);
		assert(begin != string::npos);
		char* str_end;
		unsigned int src_rank = static_cast<unsigned int>(strtoul((*it).substr(begin + 1, end - begin - 1).c_str(), &str_end, 10));
		begin = (*it).find("_", end + 1);
		end = (*it).find(".", begin + 1);
		unsigned int dst_rank = static_cast<unsigned int>(strtoul((*it).substr(begin + 1, end - begin - 1).c_str(), &str_end, 10));
		string src_filename = string(path) + string("/") + string(prefix) + string("_") + to_string(src_rank) + string("_to_") + to_string(dst_rank) + string(".npy");
		string dst_filename = string(path) + string("/") + string(comp_prefix) + string("_") + to_string(dst_rank) + string("_from_") + to_string(src_rank) + string(".npy");

		//load it into a new array
		cnpy::NpyArray src_arr = cnpy::npy_load(src_filename);
		unsigned int* sending_nids = src_arr.data<unsigned int>();
		assert(src_arr.word_size == sizeof(unsigned int));
		assert(src_arr.shape.size() == 1);

		cnpy::NpyArray dst_arr = cnpy::npy_load(dst_filename);
		unsigned int* receiving_nids = dst_arr.data<unsigned int>();
		assert(dst_arr.word_size == sizeof(unsigned int));
		assert(dst_arr.shape.size() == 1);

		assert(src_arr.shape == dst_arr.shape);
		out << "source rank (" << src_rank << ") to target rank (" << dst_rank << "):" << endl;
		unsigned int i = 0;
		for(; i < src_arr.shape[0]; i++)
		{
			if(sending_nids[i] != receiving_nids[i])
			{
				out << "the " << i << " exchange nids is not equal" << endl;
				break;
			}
		}

		if( i == src_arr.shape[0])
			out << "the " << i << " exchange nids is same" << endl;
	}
}


template<typename T>
void read_samples_from_preset(const char* filename,
							vector<shared_ptr<DataAllocator<T>>>& samples)
{
	//load it into a new array
	cnpy::NpyArray arr = cnpy::npy_load(filename); 
	T* datas = arr.data<T>();
	assert(arr.word_size == sizeof(T));
	assert(arr.shape.size() == 2 && arr.shape[0] > 0);
	samples.resize(arr.shape[0]);
	for(unsigned int i = 0; i < samples.size(); i++)
	{
		samples[i] = make_shared<DataAllocator<T>>(sizeof(T) * arr.shape[1]);
		memcpy(samples[i]->mutable_cpu_data(), datas + i * arr.shape[1], samples[i]->size());
	}
}


template shared_ptr<BrainBlock<float, float2>> init_brain_block<float, float2>(const char* filename,
																			const float delta_t,
																			const unsigned int bid,
																			const unsigned int gid);

template void check_params<float, float2>(BrainBlock<float, float2>* block,
										const char* filename);

template void check_params<float, float2>(const char* spath,
										const char* mpath,
										const unsigned int blocks);


template void read_samples_from_preset<float>(const char* filename,
							vector<shared_ptr<DataAllocator<float>>>& samples);

template void check_result<float, float2>(BrainBlock<float, float2>* block,
										const char* filename,
										const unsigned int iter,
										ostream& out);

template void check_result<float, float2>(BrainBlock<float, float2>* block,
										const unsigned int iter,
										ostream& out);

template void check_result<float, float2>(const char* mb_filepath,
										const unsigned int blks,
										const char* sb_filepath,
										ostream& out);

}//namespace istbi
