#include <thrust/detail/type_traits.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/find.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <fstream>
#include <string>
#include "configuration.hpp"
#include "common.hpp"
#include "device_function.hpp"
#include "util/transpose.hpp"
#include "util/cnpy.h"

namespace dtb {

#define MAX_ELEMS (64 * 1024 * 1024)

template<typename T>
static __global__ void merge_kernel(const T* highs,
									const unsigned int* lows,
									const unsigned int n,
									unsigned long long* vals)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
  		unsigned long long val = (static_cast<unsigned long long>(highs[i]) << 32);
		val |= lows[i];
  		vals[i] = val;
	}
}

template<typename T>
static void merge(const T* highs,
					const unsigned int* lows,
					const unsigned int n,
					thrust::device_vector<unsigned long long>& d_vals)
{
	thrust::device_vector<T> d_highs;
	thrust::device_vector<unsigned int> d_lows;
	unsigned int offset = 0;

	if(n > MAX_ELEMS)
	{
		d_highs.resize(MAX_ELEMS);
		d_lows.resize(MAX_ELEMS);
	}
	else
	{
		d_highs.resize(n);
		d_lows.resize(n);
	}
	
	do{
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_highs.data()), highs + offset, size * sizeof(T), hipMemcpyHostToDevice));
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_lows.data()), lows + offset, size * sizeof(unsigned int), hipMemcpyHostToDevice));
		merge_kernel<T><<<
					dim3(divide_up<unsigned int>(size, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					0>>>(
					thrust::raw_pointer_cast(d_highs.data()),
					thrust::raw_pointer_cast(d_lows.data()),
					size,
					thrust::raw_pointer_cast(d_vals.data()) + offset);
	
		HIP_POST_KERNEL_CHECK("merge_kernel");
		HIP_CHECK(hipDeviceSynchronize());
		offset += size;
	}while(offset < n);
	assert(offset == n);
}

template<typename T>
static void merge(const T* highs,
					const unsigned int* lows,
					const unsigned int n,
					thrust::host_vector<unsigned long long>& h_vals)
{
	thrust::device_vector<unsigned long long> d_vals(n);
	merge<T>(highs,
			lows,
			n,
			d_vals);
	HIP_CHECK(hipMemcpy(h_vals.data(), thrust::raw_pointer_cast(d_vals.data()), n * sizeof(unsigned long long), hipMemcpyDeviceToHost));
}


template<typename T>
static __global__ void split_kernel(const unsigned long long* vals,
									const unsigned int n,
									T* highs,
									unsigned int* lows)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
  		unsigned long long val = vals[i];
		highs[i] = static_cast<T>(val >> 32);
		lows[i] = static_cast<unsigned int>(val & 0xffffffffu);
	}
}

template<typename T>
static void split(const thrust::device_vector<unsigned long long>& d_vals,
				const unsigned int n,
				T* highs,
				unsigned int* lows)

{
	assert(n <= d_vals.size());
	thrust::device_vector<T> d_highs;
	thrust::device_vector<unsigned int> d_lows;

	unsigned int offset = 0;
	if(n > MAX_ELEMS)
	{
		d_highs.resize(MAX_ELEMS);
		d_lows.resize(MAX_ELEMS);
	}
	else
	{
		d_highs.resize(n);
		d_lows.resize(n);
	}
	
	do{
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		split_kernel<<<
					dim3(divide_up<unsigned int>(size, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					0>>>(
					thrust::raw_pointer_cast(d_vals.data()) + offset,
					size,
					thrust::raw_pointer_cast(d_highs.data()),
					thrust::raw_pointer_cast(d_lows.data()));
	
		HIP_POST_KERNEL_CHECK("split_kernel");
		HIP_CHECK(hipMemcpy(highs + offset, thrust::raw_pointer_cast(d_highs.data()), size * sizeof(T), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(lows + offset, thrust::raw_pointer_cast(d_lows.data()), size * sizeof(unsigned int), hipMemcpyDeviceToHost));
		offset += size;
	}while(offset < n);
	assert(offset == n);
}

template<typename T>
static void split(const thrust::host_vector<unsigned long long>& h_vals,
				const unsigned int n,
				T* highs,
				unsigned int* lows)
{
	thrust::device_vector<unsigned long long> d_vals(n);
	HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_vals.data()), h_vals.data(), n * sizeof(unsigned long long), hipMemcpyHostToDevice));
	split<T>(d_vals,
			n,
			highs,
			lows);
}

template<typename T>
static void gather_values(const thrust::host_vector<unsigned int>& h_maps,
						const unsigned int n,
						T* vals)
{
	assert(n <= h_maps.size());
	thrust::device_vector<T> d_vals(n);
	HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_vals.data()), vals, n * sizeof(T), hipMemcpyHostToDevice));

	unsigned int offset = 0;
	thrust::device_vector<unsigned int> d_maps;
	thrust::device_vector<T> d_outputs;
	
	if(n > MAX_ELEMS)
	{
		d_maps.resize(MAX_ELEMS);
		d_outputs.resize(MAX_ELEMS);
	}
	else
	{
		d_maps.resize(n);
		d_outputs.resize(n);
	}
	
	do{
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_maps.data()), h_maps.data() + offset, size * sizeof(unsigned int), hipMemcpyHostToDevice));
		thrust::gather(d_maps.begin(), d_maps.begin() + size, d_vals.begin(), d_outputs.begin());
		HIP_CHECK(hipMemcpy(vals + offset, thrust::raw_pointer_cast(d_outputs.data()), size * sizeof(T), hipMemcpyDeviceToHost));
		offset += size;
	}while(offset < n);
}

template<typename T>
static void sort_by_keys(thrust::device_vector<T>& d_keys,
							thrust::device_vector<unsigned int>& d_vals,
							const unsigned int n,
							T* keys = nullptr,
							unsigned int* vals = nullptr)
{
	if(d_keys.empty() || d_vals.empty())
	{
		return;
	}
	
	assert(n <= d_keys.size() && n <= d_vals.size());
	
	thrust::stable_sort_by_key(d_keys.begin(), d_keys.begin() + n, d_vals.begin());
	if(nullptr != keys)
	{
		HIP_CHECK(hipMemcpy(keys, thrust::raw_pointer_cast(d_keys.data()), n * sizeof(T), hipMemcpyDeviceToHost));
	}

	if(nullptr != vals)
	{
		HIP_CHECK(hipMemcpy(vals, thrust::raw_pointer_cast(d_vals.data()), n * sizeof(unsigned int), hipMemcpyDeviceToHost));
	}
}


template<typename T, bool sequenced>
static void sort_by_keys(const unsigned int n,
							T* keys,
							unsigned int* vals)
{
	if(nullptr == keys || nullptr == vals)
	{
		return;
	}
	
	thrust::device_vector<T> d_keys(n);
	thrust::device_vector<unsigned int> d_vals;

	HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys, n * sizeof(T), hipMemcpyHostToDevice));

	if(sequenced)
	{
		d_vals.resize(n);
		thrust::sequence(d_vals.begin(), d_vals.end());
	}
	
	if(!thrust::is_sorted(d_keys.begin(), d_keys.end()))
	{
		if(!sequenced)
		{
			d_vals.resize(n);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_vals.data()), vals, n * sizeof(unsigned int), hipMemcpyHostToDevice));
		}
		sort_by_keys<T>(d_keys, d_vals, n, keys, vals);
	}
	else if(sequenced)
	{
		HIP_CHECK(hipMemcpy(vals, thrust::raw_pointer_cast(d_vals.data()), n * sizeof(unsigned int), hipMemcpyDeviceToHost));
	}
}

template<typename T, bool sequenced>
static void sort_by_keys(const unsigned int n,
							T* highs,
							unsigned int* lows,
							unsigned int* vals)
{
	if(nullptr == highs || nullptr == lows || nullptr == vals)
	{
		return;
	}
	
	thrust::device_vector<unsigned long long> d_keys(n);
	merge<T>(highs, lows, n, d_keys);
	
	thrust::device_vector<unsigned int> d_vals(n);
	if(sequenced)
		thrust::sequence(d_vals.begin(), d_vals.end());
	else
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_vals.data()), vals, n * sizeof(unsigned int), hipMemcpyHostToDevice));

	sort_by_keys<unsigned long long>(d_keys, d_vals, n, nullptr, vals);
	split<T>(d_keys, n, highs, lows);
}

template<typename T, bool sequenced>
static void sort_by_keys(const thrust::host_vector<unsigned int>& h_rowptrs,
							T* highs,
							unsigned int* lows,
							thrust::host_vector<unsigned int>& h_maps)
{
	const unsigned int bins = h_rowptrs.size() - 1;
	for(unsigned int idx = 0; idx < bins;)
	{
		unsigned int n = 0;
		unsigned int num = 0;
		unsigned int offset = h_rowptrs[idx];
		
		for(unsigned int i = idx; i < bins; i++)
		{
			unsigned int count = h_rowptrs[i + 1] - h_rowptrs[i];
			if(n > 0 && (n + count) > MAX_ELEMS)
				break;

			n += count;
			num++;
			idx++;
		}

		//only single block
		if(1 == num)
		{
			sort_by_keys<unsigned int, sequenced>(n,
											lows + offset,
											h_maps.data() + offset);
		}
		else
		{
			sort_by_keys<T, sequenced>(n,
									highs + offset,
									lows + offset,
									h_maps.data() + offset);
		}
	}

}


template<typename T, bool saving_key>
static unsigned int unique_by_keys(const unsigned int n,
									T* keys,
									thrust::host_vector<unsigned int>& h_vals,
									const unsigned int init = 0)
{
	unsigned int offset = 0;
	unsigned int total_bins = 0;
	thrust::device_vector<T> d_keys;
	h_vals.clear();
	assert(h_vals.empty());
	
	if(n > MAX_ELEMS)
	{
		d_keys.resize(MAX_ELEMS);
	}
	else
	{
		d_keys.resize(n);
	}
	
	do{
		unsigned int bins_offset = 0;
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys + offset, size * sizeof(T), hipMemcpyHostToDevice));
		unsigned int bins = thrust::inner_product(d_keys.begin(), d_keys.begin() + size - 1,
	                             d_keys.begin() + 1,
	                             1,
	                             thrust::plus<unsigned int>(),
	                             thrust::not_equal_to<T>());
		assert(bins >= 1);

		thrust::device_vector<T> d_unique_keys(bins);
		thrust::device_vector<unsigned int> d_vals(bins + 1);

		// compact find the end of each bin of values
		thrust::reduce_by_key(d_keys.begin(), d_keys.begin() + size,
		                    thrust::constant_iterator<unsigned int>(1),
		                    d_unique_keys.begin(),
		                    d_vals.begin());
		assert(thrust::unique(d_unique_keys.begin(), d_unique_keys.end()) == d_unique_keys.end());

		if(!h_vals.empty())
		{
			thrust::exclusive_scan(d_vals.begin(), d_vals.end(), d_vals.begin(), h_vals.back());
			if(keys[offset] != keys[offset - 1])
			{
				bins_offset++;
			}

			unsigned int last_bins = total_bins;
			total_bins += bins + bins_offset - 1;
			h_vals.resize(total_bins + 1);
		
			HIP_CHECK(hipMemcpy(h_vals.data() + last_bins + bins_offset, thrust::raw_pointer_cast(d_vals.data()) + 1, (d_vals.size() - 1) * sizeof(unsigned int), hipMemcpyDeviceToHost));
		
			if(saving_key && (total_bins > last_bins))
			{
				HIP_CHECK(hipMemcpy(keys + last_bins + bins_offset - 1, thrust::raw_pointer_cast(d_unique_keys.data()), d_unique_keys.size() * sizeof(T), hipMemcpyDeviceToHost));
			}
		}
		else
		{
			total_bins += bins;
			h_vals.resize(total_bins + 1);
			thrust::exclusive_scan(d_vals.begin(), d_vals.end(), d_vals.begin(), init);
			HIP_CHECK(hipMemcpy(h_vals.data(), thrust::raw_pointer_cast(d_vals.data()), d_vals.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
			if(saving_key)
			{
				HIP_CHECK(hipMemcpy(keys, thrust::raw_pointer_cast(d_unique_keys.data()), d_unique_keys.size() * sizeof(T), hipMemcpyDeviceToHost));
			}
		}

		offset += size;
	}while(offset < n);

	assert(offset == n && h_vals.back() == (n + init));
	#if DEBUG
	if(saving_key)
	{
		if(total_bins > d_keys.size())
		{
			d_keys.resize(total_bins);
		}

		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys, total_bins* sizeof(T), hipMemcpyHostToDevice));
		assert(thrust::unique(d_keys.begin(), d_keys.begin() + total_bins) == (d_keys.begin() + total_bins));
	}
	#endif
	
	return total_bins;
}

template<typename T>
static void adjust_relative_conn_bid(const T bid,
										const unsigned int conns,
										T* conn_bids)
{
	unsigned int offset = 0;
	thrust::device_vector<T> d_conn_bids;
	if(conns > MAX_ELEMS)
	{
		d_conn_bids.resize(MAX_ELEMS);
	}
	else
	{
		d_conn_bids.resize(conns);
	}
	
	thrust::constant_iterator<T> constant(bid);
	thrust::plus<T> op;
	
	do{
		unsigned int size = conns - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_conn_bids.data()), conn_bids + offset, size * sizeof(T), hipMemcpyHostToDevice));
		thrust::transform(d_conn_bids.begin(), d_conn_bids.begin() + size, constant, d_conn_bids.begin(), op);
		HIP_CHECK(hipMemcpy(conn_bids + offset, thrust::raw_pointer_cast(d_conn_bids.data()), size * sizeof(T), hipMemcpyDeviceToHost));
		offset += size;
	}while(offset < conns);
	assert(offset == conns);
}

static unsigned int parse_single_block(const unsigned int conns,
										unsigned int* neuron_ids,
										unsigned int* conn_neuron_ids,
										thrust::host_vector<unsigned int>& h_maps,
										thrust::host_vector<unsigned int>& h_rowptrs)
{
	sort_by_keys<unsigned int, false>(conns,
									conn_neuron_ids,
									h_maps.data());

	gather_values<unsigned int>(h_maps, conns, neuron_ids);
	assert(thrust::is_sorted(conn_neuron_ids, conn_neuron_ids + conns));
	return  unique_by_keys<unsigned int, true>(conns,
											conn_neuron_ids,
											h_rowptrs);
}

template<typename T>
static unsigned int parse_multi_block(const unsigned int conns,
										unsigned int* neuron_ids,
										T* conn_block_ids,
										unsigned int* conn_neuron_ids,
										thrust::host_vector<unsigned int>& h_maps,
										thrust::host_vector<unsigned int>& h_rowptrs)
{	
	sort_by_keys<T, false>(h_rowptrs,
							conn_block_ids,
							conn_neuron_ids,
							h_maps);

	gather_values<unsigned int>(h_maps, conns, neuron_ids);

	thrust::host_vector<unsigned long long> h_keys(conns);
	merge<T>(conn_block_ids,
			conn_neuron_ids,
			conns,
			h_keys);

	assert(thrust::is_sorted(h_keys.begin(), h_keys.end()));
	unsigned int bins = unique_by_keys<unsigned long long, true>(conns,
													h_keys.data(),
													h_rowptrs);
	split<T>(h_keys, bins, conn_block_ids, conn_neuron_ids);
	
	return bins;
}

template<typename T>
static void parse_inputs_from_numpy(const T block_id,
								const unsigned int n,
								unsigned int* timestamps,
								unsigned int* neuron_ids,
								InputSpike& input)
{
	
	thrust::host_vector<unsigned int> h_rowptrs;
	{
		thrust::device_vector<unsigned int> d_keys(n);
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), timestamps, n * sizeof(unsigned int),hipMemcpyHostToDevice));
		if(!thrust::is_sorted(d_keys.begin(), d_keys.end()))
		{
			thrust::host_vector<unsigned int> h_maps(n);
			sort_by_keys<unsigned int, true>(n,
											timestamps,
											h_maps.data());
			
			gather_values<unsigned int>(h_maps,
										n,
										neuron_ids);
		}
	}

	assert(thrust::is_sorted(timestamps, timestamps + n));
	const unsigned int bins = unique_by_keys<unsigned int, true>(n, timestamps, h_rowptrs);
	std::cout << "input have " << bins << " timestamps" << std::endl;
	assert(h_rowptrs.size() == (bins + 1));
	assert(h_rowptrs.back() == n);
	
	input.input_timestamps = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * bins, false);
	input.input_rowptrs = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * h_rowptrs.size(), false);
	input.input_colinds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * n);
	memcpy(input.input_timestamps->mutable_cpu_data(), timestamps, input.input_timestamps->size());
	#if 0
	{
		thrust::host_vector<unsigned int> ts(bins);
		unsigned int i = 0;
		unsigned int val = -1;
		unsigned int offset = 0;
		for(; i < n; i++)
		{
			if(val != timestamps[i])
			{
				val = timestamps[i];
				ts[offset++] = val;
				assert(offset <= bins);
			}
		}
		assert(i == n && offset == bins);
		for(i = 0; i < bins; i++)
			assert(ts[i] == input.input_timestamps->cpu_data()[i]);
	}
	#endif
	memcpy(input.input_rowptrs->mutable_cpu_data(), &h_rowptrs[0], input.input_rowptrs->size());
	HIP_CHECK(hipMemcpy(input.input_colinds->mutable_gpu_data(), neuron_ids, input.input_colinds->size(), hipMemcpyDeviceToHost));
}

template<typename T, typename T2>
static void parse_outdegree_from_numpy(const T block_id,
								const std::string& filename,
								unsigned int& conns,
								unsigned int& conn_bins,
								unsigned int& same_bid_count,
								unsigned int& same_bid_begin,
								unsigned int& same_bid_end,
								thrust::host_vector<unsigned int>& h_rowptrs,
								thrust::host_vector<unsigned int>& h_maps,
								ConnectionTable<T, T2>& tab)
{
	cnpy::NpyArray arr_nids = cnpy::npz_load(filename, "output_neuron_idx");
	unsigned int* nids = arr_nids.data<unsigned int>();
	assert(arr_nids.shape.size() == 1);

	cnpy::NpyArray arr_conn_bids = cnpy::npz_load(filename, "input_block_idx");
	unsigned short* conn_bids = arr_conn_bids.data<unsigned short>();
	assert(arr_conn_bids.shape.size() == 1);

	cnpy::NpyArray arr_conn_nids = cnpy::npz_load(filename, "input_neuron_idx");
	unsigned int* conn_nids = arr_conn_nids.data<unsigned int>();
	assert(arr_conn_nids.shape.size() == 1);

	assert(arr_nids.shape[0] == arr_conn_bids.shape[0] &&
			arr_conn_bids.shape[0] == arr_conn_nids.shape[0]);

	same_bid_count = 0;
	same_bid_begin = 0;
	same_bid_end = 0;
	conns = arr_nids.shape[0];
	h_maps.resize(conns);
	
	adjust_relative_conn_bid<T>(block_id, conns, conn_bids);
	sort_by_keys<T, true>(conns, conn_bids, h_maps.data());
	assert(thrust::is_sorted(conn_bids, conn_bids + conns));
	
	unsigned int bins = unique_by_keys<T, false>(conns,
												conn_bids,
												h_rowptrs);

	if(1 == bins)
	{
		conn_bins = parse_single_block(conns,
									nids,
									conn_nids,
									h_maps,
									h_rowptrs);
		if(conn_bids[0] == block_id)
		{
			same_bid_end = conn_bins;
		}
	}
	else
	{
		gather_values<unsigned int>(h_maps, conns, conn_nids);
		conn_bins = parse_multi_block<T>(conns,
									nids,
									conn_bids,
									conn_nids,
									h_maps,
									h_rowptrs);
		
		thrust::device_vector<T> d_conn_bids(conn_bins);
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_conn_bids.data()), conn_bids, conn_bins * sizeof(T), hipMemcpyHostToDevice));
		auto it = thrust::find(d_conn_bids.begin(), d_conn_bids.end(), block_id);
		if(it != d_conn_bids.end())
		{
			assert(it == thrust::lower_bound(d_conn_bids.begin(), d_conn_bids.end(), block_id));
			same_bid_begin = it - d_conn_bids.begin();
			it = thrust::upper_bound(d_conn_bids.begin(), d_conn_bids.end(), block_id);
			same_bid_end = it - d_conn_bids.begin();
		}
	}

	assert(0 == h_rowptrs[0] && h_rowptrs[conn_bins] == conns);
	same_bid_count = same_bid_end - same_bid_begin;

	if(conn_bins > same_bid_count)
	{
		unsigned int count = conn_bins - same_bid_count;
		tab.outer_rowptrs = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * (count + 1));
		thrust::device_vector<T> d_outer_conn_bids(count);
		thrust::device_vector<unsigned int> d_outer_conn_nids(count);
		count = h_rowptrs[conn_bins] - h_rowptrs[same_bid_end] + h_rowptrs[same_bid_begin];
		tab.outer_colinds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * count);
		
		if(same_bid_begin)
		{
			HIP_CHECK(hipMemcpy((tab.outer_rowptrs)->mutable_gpu_data(), &h_rowptrs[0], (same_bid_begin + 1) * sizeof(unsigned int), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy((tab.outer_colinds)->mutable_gpu_data(), nids, h_rowptrs[same_bid_begin] * sizeof(unsigned int), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_bids.data()), conn_bids, same_bid_begin * sizeof(T), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_nids.data()), conn_nids, same_bid_begin * sizeof(unsigned int), hipMemcpyHostToDevice));
		}
		
		if(same_bid_end < conn_bins)
		{
			if(0 < same_bid_end)
			{
				thrust::device_vector<unsigned int> d_rowptrs((conn_bins - same_bid_end) + 1);
				HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_rowptrs.data()), &h_rowptrs[same_bid_end], d_rowptrs.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
				thrust::constant_iterator<unsigned int> constant(h_rowptrs[same_bid_end] - h_rowptrs[same_bid_begin]);
				thrust::minus<unsigned int> op;
				thrust::transform(d_rowptrs.begin(), d_rowptrs.end(), constant, thrust::device_pointer_cast((tab.outer_rowptrs)->mutable_gpu_data()) + same_bid_begin, op);
			}
			else
			{
				assert(same_bid_begin == 0 && same_bid_end == 0);
				HIP_CHECK(hipMemcpy((tab.outer_rowptrs)->mutable_gpu_data(), &h_rowptrs[0], h_rowptrs.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
			}

			HIP_CHECK(hipMemcpy((tab.outer_colinds)->mutable_gpu_data() + h_rowptrs[same_bid_begin], nids + h_rowptrs[same_bid_end], (h_rowptrs.back() - h_rowptrs[same_bid_end]) * sizeof(unsigned int), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_bids.data()) + same_bid_begin, conn_bids + same_bid_end, (conn_bins - same_bid_end) * sizeof(T), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_nids.data()) + same_bid_begin, conn_nids + same_bid_end, (conn_bins - same_bid_end) * sizeof(unsigned int), hipMemcpyHostToDevice));
		}

		{
			thrust::device_vector<T> d_keys;
			thrust::device_vector<unsigned int> d_counts;
			unsigned int bins = thrust::inner_product(d_outer_conn_bids.begin(), d_outer_conn_bids.end() - 1,
	                                 d_outer_conn_bids.begin() + 1,
	                                 1,
	                                 thrust::plus<unsigned int>(),
	                                 thrust::not_equal_to<T>());

			 // resize histogram storage
			 d_keys.resize(bins);
			 d_counts.resize(bins + 1);
			 tab.outer_conn_bids.resize(bins);
			 tab.outer_conn_inds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * (bins + 1));
			 tab.outer_conn_nids = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * d_outer_conn_nids.size(), false);

			// compact find the end of each bin of values
			thrust::reduce_by_key(d_outer_conn_bids.begin(), d_outer_conn_bids.end(),
			                    thrust::constant_iterator<unsigned int>(1),
			                    d_keys.begin(),
			                    d_counts.begin());
			
			thrust::exclusive_scan(d_counts.begin(),
								d_counts.end(),
								d_counts.begin());

			HIP_CHECK(hipMemcpy(tab.outer_conn_bids.data(), thrust::raw_pointer_cast(d_keys.data()), d_keys.size() * sizeof(T), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy((tab.outer_conn_inds)->mutable_cpu_data(), thrust::raw_pointer_cast(d_counts.data()), d_counts.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
			assert(((tab.outer_conn_inds)->cpu_data()[bins] - (tab.outer_conn_inds)->cpu_data()[0]) == d_outer_conn_nids.size());
			HIP_CHECK(hipMemcpy((tab.outer_conn_nids)->mutable_cpu_data(), thrust::raw_pointer_cast(d_outer_conn_nids.data()), d_outer_conn_nids.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
		}
	}

	if(same_bid_count)
	{
		tab.inner_rowptrs = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * (same_bid_count + 1));
		unsigned int count = h_rowptrs[same_bid_end] - h_rowptrs[same_bid_begin];
		tab.inner_colinds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * count);
		tab.inner_conninds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * same_bid_count);
		
		if(same_bid_begin)
		{
			thrust::device_vector<unsigned int> d_rowptrs(same_bid_count + 1);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_rowptrs.data()), &h_rowptrs[same_bid_begin], d_rowptrs.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
			thrust::constant_iterator<unsigned int> constant(h_rowptrs[same_bid_begin]);
			thrust::minus<unsigned int> op;
			thrust::transform(d_rowptrs.begin(), d_rowptrs.end(), constant, thrust::device_pointer_cast((tab.inner_rowptrs)->mutable_gpu_data()), op);
		}
		else
		{
			HIP_CHECK(hipMemcpy((tab.inner_rowptrs)->mutable_gpu_data(), &h_rowptrs[0], (same_bid_count + 1) * sizeof(unsigned int), hipMemcpyHostToDevice));
		}
		HIP_CHECK(hipMemcpy((tab.inner_colinds)->mutable_gpu_data(), nids + h_rowptrs[same_bid_begin], count * sizeof(unsigned int), hipMemcpyHostToDevice));
		HIP_CHECK(hipMemcpy((tab.inner_conninds)->mutable_gpu_data(), conn_nids + same_bid_begin, (same_bid_end - same_bid_begin) * sizeof(unsigned int), hipMemcpyHostToDevice));
	}
}

template<typename T, typename T2, typename C>
CONFIG_BLOCK_TYPE parse_conn_table_from_numpy(const C block_id,
								const std::string& filename,
								InputSpike& input,
								ConnectionTable<C, T2>& tab)
{
	if(cnpy::npz_find(filename, "src_timestamp"))
	{
		cnpy::NpyArray arr_ts = cnpy::npz_load(filename, "src_timestamp");
		cnpy::NpyArray arr_nids = cnpy::npz_load(filename, "src_neuron_idx");
		unsigned int* timestamps = arr_ts.data<unsigned int>();
		unsigned int* nids = arr_nids.data<unsigned int>();
		assert(arr_ts.shape[0] == arr_nids.shape[0]);
		parse_inputs_from_numpy<C>(block_id, arr_nids.shape[0], timestamps, nids, input);
		return CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_INPUT;
	}

	unsigned int conns;
	unsigned int conn_bins;
	unsigned int same_bid_count = 0;
	unsigned int same_bid_begin = 0;
	unsigned int same_bid_end = 0;
	thrust::host_vector<unsigned int> h_rowptrs;
	thrust::host_vector<unsigned int> h_maps;

	parse_outdegree_from_numpy<C, T2>(block_id,
							filename,
							conns,
							conn_bins,
							same_bid_count,
							same_bid_begin,
							same_bid_end,
							h_rowptrs,
							h_maps,
							tab);
		
	cnpy::NpyArray arr_conn_kinds = cnpy::npz_load(filename, "input_channel_offset");
	unsigned char* conn_kinds = arr_conn_kinds.data<unsigned char>();
	assert(arr_conn_kinds.shape.size() == 1 &&
		arr_conn_kinds.shape[0] == conns);

	cnpy::NpyArray arr_weight = cnpy::npz_load(filename, "weight");
	T2* weights = reinterpret_cast<T2*>(arr_weight.data<T>());
	assert(arr_weight.shape.size() == 2 &&
		arr_weight.shape[0] == conns &&
		arr_weight.shape[1] == 2);

	gather_values<unsigned char>(h_maps, conns, conn_kinds);
	gather_values<T2>(h_maps, conns, weights);
	h_maps.clear();
		
	if(conn_bins > same_bid_count)
	{
		unsigned int count = conn_bins - same_bid_count;
		count = h_rowptrs[conn_bins] - h_rowptrs[same_bid_end] + h_rowptrs[same_bid_begin];
		tab.outer_vals = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * count);
		tab.outer_connkinds = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), sizeof(unsigned char) * count);
		
		if(same_bid_begin)
		{
			HIP_CHECK(hipMemcpy((tab.outer_vals)->mutable_gpu_data(), weights, h_rowptrs[same_bid_begin] * sizeof(T2), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy((tab.outer_connkinds)->mutable_gpu_data(), conn_kinds, h_rowptrs[same_bid_begin] * sizeof(unsigned char), hipMemcpyHostToDevice));
		}
		
		if(same_bid_end < conn_bins)
		{
			HIP_CHECK(hipMemcpy((tab.outer_vals)->mutable_gpu_data() + h_rowptrs[same_bid_begin], weights + h_rowptrs[same_bid_end], (h_rowptrs.back() - h_rowptrs[same_bid_end]) * sizeof(T2), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy((tab.outer_connkinds)->mutable_gpu_data() + h_rowptrs[same_bid_begin], conn_kinds + h_rowptrs[same_bid_end], (h_rowptrs.back() - h_rowptrs[same_bid_end]) * sizeof(unsigned char), hipMemcpyHostToDevice));
		}
	}

	if(same_bid_count)
	{
		unsigned int count = h_rowptrs[same_bid_end] - h_rowptrs[same_bid_begin];
		tab.inner_vals = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * count);
		tab.inner_connkinds = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), sizeof(unsigned char) * count);
		HIP_CHECK(hipMemcpy((tab.inner_vals)->mutable_gpu_data(), weights + h_rowptrs[same_bid_begin], (h_rowptrs[same_bid_end] - h_rowptrs[same_bid_begin])* sizeof(T2), hipMemcpyHostToDevice));
		HIP_CHECK(hipMemcpy((tab.inner_connkinds)->mutable_gpu_data(), conn_kinds + h_rowptrs[same_bid_begin], (h_rowptrs[same_bid_end] - h_rowptrs[same_bid_begin])* sizeof(unsigned char), hipMemcpyHostToDevice));
	}

	return CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_NORMAL;
}

template<typename T>
struct cap_reciprocal

{
  __host__ __device__ inline
  T operator()(const T& a) const
  {
    return (T)1 / a;
  }
};


template<typename T, typename T2>
struct type_convertion
{
  __host__ __device__
  T2 operator()(const T& x, const T& y) const
  {
  	T2 result;
  	result.x = x;
  	result.y = y;
    return result;
  }
};

template<typename T>
struct tao_transformation
{
  __host__ __device__
  T operator()(const T& x, const T& y) const
  {
  	T result;
  	result = texp<T>(y / x);
    return result;
  }
};

template<typename T, typename T2>
struct floor_comparision
{
  __host__ __device__
  T operator()(const T2& val) const
  {
    return (val.x <= val.y) ? val.x : val.y;
  }
};

template<typename T>
static __global__ void init_index_kernel(const T* inputs,
										const unsigned int n,
					                    unsigned int* indices)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int index = 0xffffffff;
		if(inputs[i] != static_cast<T>(0))
			index = i;

		indices[i] = index;
	}
}

template<typename T>
static void flat_non_zero(thrust::device_vector<T>& inputs,
							thrust::device_vector<unsigned int>& indices,
							unsigned int& n)
{
	init_index_kernel<T><<< 
					dim3(divide_up<unsigned int>(inputs.size(), HIP_THREADS_PER_BLOCK)), 
					dim3(HIP_THREADS_PER_BLOCK), 
					0, 
					0>>>(
					thrust::raw_pointer_cast(inputs.data()),
					inputs.size(),
					thrust::raw_pointer_cast(indices.data()));
																									
	HIP_POST_KERNEL_CHECK("init_index_kernel");
	HIP_CHECK(hipDeviceSynchronize());
	thrust::sort_by_key(indices.begin(), indices.end(), inputs.begin());
	n = (thrust::find(indices.begin(), indices.end(), 0xffffffff) - indices.begin());
}

static __global__ void stat_exclusive_count_kernel(const unsigned int* exclusive_bids,
													const unsigned int* bcounts,
													const unsigned int m,
													const unsigned int* bids,
													const unsigned int n,
								                    unsigned int* exclusive_counts)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < m; i += gridSize)
	{
		unsigned int bid = exclusive_bids[i];
		const unsigned int* iter = thrust::find(thrust::device, bids, bids + n, bid);
		unsigned int idx = iter - bids;
		if(idx != n)
		{
			exclusive_counts[idx] = bcounts[i];
		}
	}
}

static void stat_exclusive_count(const unsigned int* exclusive_bids,
									const unsigned int* bcounts,
									const unsigned int m,
									const unsigned int* bids,
									const unsigned int n,
				                    unsigned int* exclusive_counts)
{
	stat_exclusive_count_kernel<<< 
								dim3(divide_up<unsigned int>(m, HIP_THREADS_PER_BLOCK)), 
								dim3(HIP_THREADS_PER_BLOCK), 
								0, 
								0>>>(
								exclusive_bids,
								bcounts,
								m,
								bids,
								n,
								exclusive_counts);
	HIP_POST_KERNEL_CHECK("stat_exclusive_count_kernel");
	HIP_CHECK(hipDeviceSynchronize());
}


template<typename T, typename T2, typename C>
void parse_params_from_numpy(const C block_id,
									const std::string& filename,
									const T delta_t,
									unsigned int& neurons,
									ConfigParameter<T, T2>& params)
{
	//static unsigned int upper_size = 65535 * 32;
	unsigned int height;
	unsigned int width;
	thrust::device_vector<T> d_props;
	{
		cnpy::NpyArray arr_prop = cnpy::npz_load(filename, "property");
		T* props = arr_prop.data<T>();
		assert(arr_prop.shape.size() == 2 &&  arr_prop.shape[1] == 22);
		height = arr_prop.shape[0];
		width = arr_prop.shape[1];
		neurons = height;
		size_t total = static_cast<size_t>(height) * width;
		
		d_props.resize(total);
		//if(height <= upper_size)
		//{
			thrust::device_vector<T> d_temps(total);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_temps.data()), props, total * sizeof(T), hipMemcpyHostToDevice));
			transpose_gpu<T>(thrust::raw_pointer_cast(d_temps.data()),
							height,
							width,
							thrust::raw_pointer_cast(d_props.data()));
			HIP_CHECK(hipDeviceSynchronize());
		//}
		/*else
		{
			thrust::host_vector<T> h_temps(total);
			transpose_cpu<T>(props, height, width, h_temps.data());
			thrust::copy(h_temps.begin(), h_temps.end(), d_props.begin());
		}
		*/
	}

	unsigned int soffset = 0;
	unsigned int eoffset = height;
	{
		params.noise_rates = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
		thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.noise_rates)->mutable_gpu_data()));
	}
	
	soffset += height;
	eoffset += height;

	thrust::device_vector<unsigned int> d_exclusive_colinds;
	{
		params.exclusive_flags = nullptr;
		thrust::device_vector<T> d_datum(height);
		thrust::device_vector<unsigned int> d_indices(height);
		thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, d_datum.begin());
		unsigned int n;
		flat_non_zero<T>(d_datum, d_indices, n);

		if(n > 0)
		{
			params.exclusive_flags = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), sizeof(unsigned char) * height);
			thrust::transform(d_props.begin() + soffset, 
							d_props.begin() + eoffset, 
							thrust::device_pointer_cast((params.exclusive_flags)->mutable_gpu_data()),
							thrust::identity<unsigned char>());
			#if DEBUG
			thrust::host_vector<unsigned int> h_indices(n);
			thrust::copy(d_indices.begin(), d_indices.begin() + n, h_indices.begin());
			thrust::copy((params.exclusive_flags)->gpu_data(), (params.exclusive_flags)->gpu_data() + height, (params.exclusive_flags)->mutable_cpu_data());
			for(unsigned int i = 0; i < height; i++)
			{
				unsigned int j = 0;
				for(; j < n; j++)
				{
					if(h_indices[j] == i)
						break;
				}

				if(j < n)
				{
					assert((params.exclusive_flags)->cpu_data()[i] == 0x01);
				}
				else
				{
					assert((params.exclusive_flags)->cpu_data()[i] == 0x00);
				}
			}
			#endif
			d_exclusive_colinds.resize(n);
			thrust::copy(d_indices.begin(), d_indices.begin() + n, d_exclusive_colinds.begin());
		}
	}

	soffset += height;
	eoffset += height;
	{
		params.i_ext_stimuli = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
		thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.i_ext_stimuli)->mutable_gpu_data()));
	}
	
	soffset += height;
	eoffset += height;
	{
		params.exclusive_counts = nullptr;
		thrust::device_vector<T> d_datum(height);
		thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, d_datum.begin());
		unsigned int bins = thrust::inner_product(d_datum.begin(),
												d_datum.end() - 1,
												d_datum.begin() + 1,
												1,
												thrust::plus<unsigned int>(),
												thrust::not_equal_to<T>());
		if(bins > 0)
		{
			thrust::device_vector<unsigned int> d_bids(bins);
			thrust::device_vector<unsigned int> d_indices(bins);
			thrust::device_vector<T> d_uniquekeys(bins);
			thrust::device_vector<unsigned int> d_counts(bins);
			
			// compact find the end of each bin of values
			thrust::reduce_by_key(d_datum.begin(),
								d_datum.end(),
			                    thrust::constant_iterator<unsigned int>(1),
			                    d_uniquekeys.begin(),
			                    d_counts.begin());

			thrust::exclusive_scan(d_counts.begin(),
								d_counts.end(),
								d_indices.begin());

			thrust::transform(d_uniquekeys.begin(), 
							d_uniquekeys.end(), 
							d_bids.begin(),
							thrust::identity<unsigned int>());

			params.subids = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * d_bids.size());
			thrust::copy(d_bids.begin(), d_bids.end(), (params.subids)->mutable_gpu_data());
			thrust::copy(d_bids.begin(), d_bids.end(), (params.subids)->mutable_cpu_data());
			params.subcounts = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * d_counts.size(), false);
			thrust::copy(d_counts.begin(), d_counts.end(), (params.subcounts)->mutable_cpu_data());
			
			params.subinfos = make_shared<DataAllocator<uint2>>(static_cast<int>(block_id + 1), sizeof(uint2) * d_indices.size());
			thrust::transform(d_indices.begin(), d_indices.end(), d_counts.begin(), thrust::device_pointer_cast((params.subinfos)->mutable_gpu_data()), type_convertion<unsigned int, uint2>());

			if(!d_exclusive_colinds.empty())
			{
				params.exclusive_counts = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * params.subids->count());
				HIP_CHECK(hipMemset((params.exclusive_counts)->mutable_gpu_data(),
									0x00,
									(params.exclusive_counts)->size()));
				thrust::device_vector<T> d_exclusive_datum(d_exclusive_colinds.size());
				
				thrust::gather(d_exclusive_colinds.begin(),
							d_exclusive_colinds.end(),
							d_datum.begin(),
							d_exclusive_datum.begin());

				bins = thrust::inner_product(d_exclusive_datum.begin(), 
											d_exclusive_datum.end() - 1,
											d_exclusive_datum.begin() + 1,
											1,
											thrust::plus<unsigned int>(),
											thrust::not_equal_to<T>());
				assert(bins > 0 && bins <= params.subids->count());

				if(bins > 0)
				{
					 // resize histogram storage	
					d_uniquekeys.resize(bins);
					d_counts.resize(bins);
					d_indices.resize(bins);
					d_bids.resize(bins);

					// compact find the end of each bin of values
					thrust::reduce_by_key(d_exclusive_datum.begin(),
										d_exclusive_datum.end(),
										thrust::constant_iterator<unsigned int>(1),
										d_uniquekeys.begin(),
										d_counts.begin());
					
					thrust::transform(d_uniquekeys.begin(), 
									d_uniquekeys.end(), 
									d_bids.begin(),
									thrust::identity<unsigned int>());

					stat_exclusive_count(thrust::raw_pointer_cast(d_bids.data()),
										thrust::raw_pointer_cast(d_counts.data()),
										bins,
										params.subids->mutable_gpu_data(),
										params.subids->count(),
										params.exclusive_counts->mutable_gpu_data());
				}
			}
		}
	}

	soffset += height;
	eoffset += height;
	params.c_membrane_reciprocals= make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	//thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.c_membrane_reciprocals)->mutable_gpu_data()));
	thrust::transform(d_props.begin() + soffset, 
					d_props.begin() + eoffset, 
					thrust::device_pointer_cast((params.c_membrane_reciprocals)->mutable_gpu_data()),
					cap_reciprocal<T>());
	
	soffset += height;
	eoffset += height;
	params.t_refs = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.t_refs)->mutable_gpu_data()));

	soffset += height;
	eoffset += height;
	params.g_leakages = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.g_leakages)->mutable_gpu_data()));

	soffset += height;
	eoffset += height;
	params.v_leakages = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.v_leakages)->mutable_gpu_data()));

	soffset += height;
	eoffset += height;
	params.v_thresholds = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.v_thresholds)->mutable_gpu_data()));

	soffset += height;
	eoffset += height;
	params.v_resets = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	thrust::copy(d_props.begin() + soffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.v_resets)->mutable_gpu_data()));

	{
		soffset += height;
		eoffset += height;
		params.g_ex_conducts = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.g_ex_conducts)->mutable_gpu_data()), type_convertion<T, T2>());
		soffset += height;
		eoffset += height;
	}

	{
		soffset += height;
		eoffset += height;
		params.g_in_conducts = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.g_in_conducts)->mutable_gpu_data()), type_convertion<T, T2>());
		soffset += height;
		eoffset += height;
	}

	{
		soffset += height;
		eoffset += height;
		params.v_ex_membranes = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.v_ex_membranes)->mutable_gpu_data()), type_convertion<T, T2>());
		soffset += height;
		eoffset += height;
	}

	{
		soffset += height;
		eoffset += height;
		params.v_in_membranes = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.v_in_membranes)->mutable_gpu_data()), type_convertion<T, T2>());
		soffset += height;
		eoffset += height;
	}

	/*
	{
		params.v_floors = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
		thrust::device_vector<T2> d_in_membranes(height);
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_in_membranes.data()), params.v_in_membranes->gpu_data(), params.v_in_membranes->size(), hipMemcpyDeviceToDevice));
		HIP_CHECK(hipMemcpy(params.v_in_membranes->mutable_cpu_data(), thrust::raw_pointer_cast(d_in_membranes.data()), params.v_in_membranes->size(), hipMemcpyDeviceToHost));
		//std::cout << "=======v_in_membranes: " << params.v_in_membranes->cpu_data()[0].x << ", " << params.v_in_membranes->cpu_data()[0].y << "=============" << std::endl;
		thrust::transform(d_in_membranes.begin(), d_in_membranes.end(), thrust::device_pointer_cast((params.v_floors)->mutable_gpu_data()), floor_comparision<T, T2>());
		std::vector<T> h_v_floors(height);
		HIP_CHECK(hipMemcpy(h_v_floors.data(), params.v_floors->gpu_data(), h_v_floors.size() * sizeof(T), hipMemcpyDeviceToHost));
		//std::cout << "=======v_floors: " << h_v_floors[0] << "=============" << std::endl;
	}
	*/

	{
		thrust::device_vector<T> d_scales(height);
		thrust::fill_n(d_scales.begin(), d_scales.size(), (-1) * delta_t);
		soffset += height;
		eoffset += height;
		
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_scales.begin(), d_props.begin() + soffset, tao_transformation<T>());
		thrust::transform(d_props.begin() + eoffset, d_props.begin() + eoffset + height, d_scales.begin(), d_props.begin() + eoffset, tao_transformation<T>());
		params.tao_ex_constants = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.tao_ex_constants)->mutable_gpu_data()), type_convertion<T, T2>());
		soffset += height;
		eoffset += height;
	
		soffset += height;
		eoffset += height;
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_scales.begin(), d_props.begin() + soffset, tao_transformation<T>());
		thrust::transform(d_props.begin() + eoffset, d_props.begin() + eoffset + height, d_scales.begin(), d_props.begin() + eoffset, tao_transformation<T>());
		params.tao_in_constants = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		thrust::transform(d_props.begin() + soffset, d_props.begin() + eoffset, d_props.begin() + eoffset, thrust::device_pointer_cast((params.tao_in_constants)->mutable_gpu_data()), type_convertion<T, T2>());
	}
}


template CONFIG_BLOCK_TYPE parse_conn_table_from_numpy<float, float2, unsigned short>(const unsigned short block_id,
																const std::string& filename,
																InputSpike& input,
																ConnectionTable<unsigned short, float2>& tab);

template void parse_params_from_numpy<float, float2, unsigned short>(const unsigned short block_id,
																	const std::string& filename,
																	const float delta_t,
																	unsigned int& neurons,
																	ConfigParameter<float, float2>& params);

}//namespace istbi
