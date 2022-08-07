#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"
#include <iostream>

/* include MTGP host helper functions */
#include <hiprand_mtgp32_host.h>

//#if defined(__HIP_PLATFORM_HCC__)
/* include MTGP pre-computed parameter sets */
#include <rocrand_mtgp32_11213.h>
//#elif defined(__HIP_PLATFORM_NVCC__)
/* include MTGP pre-computed parameter sets */
//#include <curand_mtgp32dc_p_11213.h>
//#endif

#include <rocprim/rocprim.hpp>

namespace dtb {

#define MTGP32_MAX_BLOCK_SIZE 256

template<typename T>
static __global__ void init_spike_time_kernel(const unsigned int n,
												const T val,
												T* t_actives)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
		t_actives[i] = val;
	}
}


template<typename T>
void init_spike_time_gpu(const unsigned int n,
							const T val,
							T* t_actives,
							hipStream_t stream)
{
	hipLaunchKernelGGL(HIP_KERNEL_NAME(init_spike_time_kernel<T>),
				dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
				dim3(HIP_THREADS_PER_BLOCK),
				0,
				stream,
				n,
				val,
				t_actives);
	HIP_POST_KERNEL_CHECK("init_spike_time_kernel");
}

/* Creates a new generator state given the seed. Not thread-safe. */
void create_generator_state(const unsigned long long seed, 
							hiprandStateMtgp32* states,
							mtgp32_kernel_params_t* params)
{
	HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, params));
	HIPRAND_CHECK(hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, params, MTGP32_MAX_NUM_BLOCKS, seed));
}

// Goes from (0, 1] to [0, 1). Note 1-x is not sufficient since for some floats
// eps near 0, 1-eps will round to 1.
template<typename T>
static __device__ inline T reverse_bounds(T value) 
{
  if (value == static_cast<T>(1))
  {
    return static_cast<T>(0);
  }
  
  return value;
}

template<typename T>
static __global__ void generate_uniform_samples_kernel(hiprandStateMtgp32* states,
															const unsigned int n,
															const T a,
															const T b,
															T* samples);

template<>
__global__ void generate_uniform_samples_kernel<float>(hiprandStateMtgp32* states,
															const unsigned int n,
															const float a,
															const float b,
															float* samples)

{
	const unsigned int idx = blockIdx.x * MTGP32_MAX_BLOCK_SIZE + threadIdx.x;
	const unsigned int roundSize = align_up<8, unsigned int>(n);
	
  	for(unsigned int i = idx; i < roundSize; i += MTGP32_MAX_BLOCK_SIZE * MTGP32_MAX_NUM_BLOCKS)
	{
		float x = hiprand_uniform(&states[blockIdx.x]);
		if(i < n)
		{
			float y = reverse_bounds<float>(x) * (b-a) + a; 
			samples[i] = x;
		}
	}
}

template<>
void generate_uniform_samples<float>(hiprandStateMtgp32* states,
										const unsigned int n,
										const float a,
										const float b,
										float* samples,
										hipStream_t stream)
{
	unsigned int blocks = MIN(divide_up<unsigned int>(n, MTGP32_MAX_BLOCK_SIZE), MTGP32_MAX_NUM_BLOCKS);
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(generate_uniform_samples_kernel<float>),
					dim3(blocks),
					dim3(MTGP32_MAX_BLOCK_SIZE),
					0,
					stream,
					states,
					n,
					a,
					b,
					samples);
	HIP_POST_KERNEL_CHECK("generate_uniform_samples_kernel");
}

template<typename T>
static __global__ void update_spike_kernel(hiprandStateMtgp32* states,
											const unsigned int n,
											const T* noise_rates,
											const T a,
											const T b,
											unsigned char* f_actives,
											T* samples);

template<>
__global__ void update_spike_kernel<float>(hiprandStateMtgp32* states,
											const unsigned int n,
											const float* noise_rates,
											const float a,
											const float b,
											unsigned char* f_actives,
											float* samples)
{
	const unsigned int idx = blockIdx.x * MTGP32_MAX_BLOCK_SIZE+ threadIdx.x;
	const unsigned int roundSize = align_up<8, unsigned int>(n);
	
  	for(unsigned int i = idx; i < roundSize; i += MTGP32_MAX_BLOCK_SIZE * MTGP32_MAX_NUM_BLOCKS)
	{
		float x = hiprand_uniform(&states[blockIdx.x]);
		if(i < n)
		{
			unsigned char fi = f_actives[i];
			float y = reverse_bounds<float>(x) * (b - a) + a; 
     	 	fi |= static_cast<unsigned char>(y < noise_rates[i]);
			f_actives[i] = fi;
			if(NULL != samples)
				samples[i] = y;
		}
	}
}

template<>
void update_spike_gpu<float>(hiprandStateMtgp32* states,
						const unsigned int n,
						const float* noise_rates,
						unsigned char* f_actives,
						const float a,
						const float b,
						float* samples,
						hipStream_t stream)
{
	
	unsigned int blocks = MIN(divide_up<unsigned int>(n, MTGP32_MAX_BLOCK_SIZE), MTGP32_MAX_NUM_BLOCKS);
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_spike_kernel<float>),
					dim3(blocks), 
					dim3(MTGP32_MAX_BLOCK_SIZE),
					0,
					stream,
					states,
					n,
					noise_rates,
					a,
					b,
					f_actives,
					samples);
	HIP_POST_KERNEL_CHECK("update_spike_kernel");
}

template<
	unsigned int blockSize,
	class CompareFunction = ::rocprim::less<>
>
static __global__  void update_routing_offsets_kernel(const unsigned int* routing_unions,
													const unsigned int routing_union_nums,
													const unsigned int routing_offset_nums,
													unsigned int* routing_offsets,
													CompareFunction compare_op = CompareFunction())
{
	__shared__ unsigned int s_datas[blockSize * 2];
	
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockSize + tid;
	unsigned int val;
	bool dirty = false;

	for(unsigned int i = 0; i < divide_up<unsigned int>(routing_union_nums, blockSize * 2); i++)
	{
		__syncthreads();
		unsigned int ai = tid + i * blockSize * 2;
		unsigned int bi = tid + i * blockSize * 2 + blockSize;
		s_datas[tid] = (ai < routing_union_nums)? routing_unions[ai] : 0xffffffff;
		s_datas[tid + blockSize] = (bi < routing_union_nums) ? routing_unions[bi] : 0xffffffff;
		__syncthreads();
		
		if((idx < routing_offset_nums) && !dirty)
		{
			val = routing_offsets[idx];
			const unsigned int oidx = ::rocprim::detail::lower_bound_n(s_datas, 2 * blockSize, val, compare_op);
			if(oidx != (2 * blockSize) && !compare_op(val, s_datas[oidx]))
			{
				routing_offsets[idx] = i * blockSize * 2 + static_cast<unsigned int>(oidx);
				dirty = true;
			}
		}
	}
}

void update_routing_offsets_gpu(const unsigned int* routing_unions,
									const unsigned int routing_union_nums,
									const unsigned int routing_offset_nums,
									unsigned int* routing_offsets,
									hipStream_t stream)
{
	const unsigned int blocks = divide_up<unsigned int>(routing_offset_nums, HIP_THREADS_PER_BLOCK);
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_routing_offsets_kernel<HIP_THREADS_PER_BLOCK>),
					dim3(blocks), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					routing_unions,
					routing_union_nums,
					routing_offset_nums,
					routing_offsets);
	HIP_POST_KERNEL_CHECK("compute_routing_offsets_kernel");
}

template<
	unsigned int BlockSize,
	unsigned int ItemsPerThread,
	class InputIterator,
	class OutputIterator
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU) 
void transform_kernel(
				InputIterator inputs,
				const size_t size,
				OutputIterator outputs)
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	static_assert(::rocprim::is_integral<output_type>::value, "Type IndexIterator must be integral type");
	static_assert(std::is_convertible<input_type, output_type>::value,
                      "The type OutputIterator must be such that an object of type InputIterator"
                      "can be dereferenced and then implicitly converted to OutputIterator.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr int items_per_block = BlockSize * ItemsPerThread;
	auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    unsigned int block_offset = (flat_block_id * items_per_block);

	output_type values[ItemsPerThread];
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values,
            valid_in_last_block
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = flat_id + i * BlockSize;
			if(offset < valid_in_last_block)
			{
				outputs[block_offset + offset] = static_cast<output_type>(inputs[values[i]]);
			}
		}
    }
    else
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = flat_id + i * BlockSize;
			outputs[block_offset + offset] = static_cast<output_type>(inputs[values[i]]);

		}
    }
}


template<
	unsigned int BlockSize,
	unsigned int ItemsPerThread,
	class InputIterator,
	class OutputIterator,
	class Default
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void fill_kernel(
				InputIterator inputs,
				const size_t size,
				OutputIterator outputs,
				Default default_value)
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	static_assert(::rocprim::is_integral<input_type>::value, "Type IndexIterator must be integral type");
	static_assert(std::is_convertible<Default, output_type>::value,
					  "The type OutputIterator must be such that an object of type Default "
					  "can be dereferenced and then implicitly converted to type OutputIterator.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr int items_per_block = BlockSize * ItemsPerThread;
	auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    unsigned int block_offset = (flat_block_id * items_per_block);

	input_type values[ItemsPerThread];
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            inputs + block_offset,
            values,
            valid_in_last_block
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = i * BlockSize;
			if(flat_id + offset < valid_in_last_block)
			{
				outputs[values[i]] = static_cast<output_type>(default_value);
			}
		}
    }
    else
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            inputs + block_offset,
            values
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			outputs[values[i]] = static_cast<output_type>(default_value);
		}
    }
}


template<
	unsigned int BlockSize,
	class T,
	unsigned int ItemsPerThread,
    class InputIterator,
    class IndexIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<T>
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void expand_and_reduce_kernel(
					InputIterator inputs,
					IndexIterator indice,
					const size_t size,
					OutputIterator outputs,
					T* reductions,
					const T out_of_bound = (T)0,
					const T initial_value = (T)0,
					BinaryFunction reduce_op = BinaryFunction())
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	using index_type = typename std::iterator_traits<IndexIterator>::value_type;

	static_assert(::rocprim::is_integral<index_type>::value, "Type IndexIterator must be integral type");
	static_assert(std::is_convertible<input_type, output_type>::value,
					  "The type OutputIterator must be such that an object of type InputIterator "
					  "can be dereferenced and then implicitly converted to type OutputIterator.");
	static_assert(std::is_convertible<output_type, T>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly converted to T.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr int items_per_block = BlockSize * ItemsPerThread;
	auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    unsigned int block_offset = (flat_block_id * items_per_block);

	using block_reduce_type = ::rocprim::block_reduce<
        T, BlockSize,
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
	
	union{
		output_type values[ItemsPerThread];
		index_type indice[ItemsPerThread];
	} storage;

	T reduction = out_of_bound;
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            indice + block_offset,
            storage.indice,
            valid_in_last_block
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = i * BlockSize;
			if(flat_id + offset < valid_in_last_block)
			{
				storage.values[i] = inputs[storage.indice[i]];
				reduction = reduce_op(reduction, static_cast<T>(storage.values[i]));
			}
		}

		 block_reduce_type()
            .reduce(
                reduction, // input
                reduction, // output
                reduce_op
            );
		 
		 ::rocprim::block_store_direct_striped<BlockSize>(
			 flat_id,
			 outputs + block_offset,
			 storage.values,
			 valid_in_last_block
		 );
    }
    else
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            indice + block_offset,
            storage.indice
        );

		#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
			storage.values[i] = inputs[storage.indice[i]];
			reduction = reduce_op(reduction, static_cast<T>(storage.values[i]));
        }

		// load input values into values
        block_reduce_type()
            .reduce(
                reduction, // input
                reduction, // output
                reduce_op
            );

		::rocprim::block_store_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            storage.values
        );
    }

	// Save block reduction
    if(flat_id == 0)
    {
       reductions[flat_block_id] = reduce_op(reduction, initial_value);
    }
}

template<
	unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OffsetIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<OutputIterator>::value_type>,
    class Default = typename std::iterator_traits<OutputIterator>::value_type
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_and_select_kernel(
						InputIterator inputs,
						const size_t size,
						OffsetIterator offsets,
						OutputIterator outputs,
						Default out_of_bound = Default(),
						BinaryFunction scan_op = BinaryFunction())
{	
	using offset_type = typename std::iterator_traits<OffsetIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	static_assert(std::is_convertible<offset_type, output_type>::value,
                      "The type OutputIterator must be such that an object of type OffsetIterator "
                      "can be dereferenced and then implicitly converted to OutputIterator.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr auto items_per_block = BlockSize * ItemsPerThread;
	unsigned int valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    auto block_offset = (flat_block_id * items_per_block);

   using block_load_type = ::rocprim::block_load<
        output_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose
    >;
    using block_exchange_type = ::rocprim::block_exchange<
        output_type, BlockSize, ItemsPerThread
    >;

	using block_scan_type = ::rocprim::block_scan<
        output_type, BlockSize,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;

    __shared__ union
    {
        typename block_load_type::storage_type load;
        typename block_exchange_type::storage_type exchange;
        typename block_scan_type::storage_type scan;
    } storage;

    output_type values[ItemsPerThread];
	offset_type offset = offsets[flat_block_id];

	if(offsets[flat_block_id + 1] <= offset)
		return;
	
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
	    // load input values into values
	    block_load_type()
	        .load(
	            inputs + block_offset,
	            values,
	            valid_in_last_block,
	            out_of_bound,
	            storage.load
	        );
		 
	    ::rocprim::syncthreads(); // sync threads to reuse shared memory

		block_scan_type()
			.exclusive_scan(
	        values, // input
	        values, // output
	        static_cast<output_type>(offset),
	        storage.scan,
	        scan_op
	    );

		::rocprim::syncthreads(); // sync threads to reuse shared memory

		// Save values into output array
		block_exchange_type()
			.blocked_to_striped(
			   values, 
			   values,
			   storage.exchange
			);

		::rocprim::syncthreads(); // sync threads to reuse shared memory

        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            auto _offset = i * BlockSize + flat_id;
            if(_offset < valid_in_last_block && inputs[block_offset + _offset])
            {
                outputs[values[i]] = block_offset + _offset;
            }
        }

    }
    else
    {
    	 // load input values into values
	    block_load_type()
	        .load(
	            inputs + block_offset,
	            values,
	            storage.load
	        );
		 
	    ::rocprim::syncthreads(); // sync threads to reuse shared memory

		block_scan_type()
			.exclusive_scan(
	        values, // input
	        values, // output
	        static_cast<output_type>(offset),
	        storage.scan,
	        scan_op
	    );

		::rocprim::syncthreads(); // sync threads to reuse shared memory

		// Save values into output array
		block_exchange_type()
		.blocked_to_striped(
		   values, 
		   values,
		   storage.exchange
		);

		::rocprim::syncthreads(); // sync threads to reuse shared memory
		
		#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            auto _offset = i * BlockSize + flat_id;
            if(inputs[block_offset + _offset])
            {
                 outputs[values[i]] = block_offset + _offset;
            }
        }
    }
	
}

template<
	unsigned int BlockSize,
	unsigned int ItemsPerThread,
	class InputIterator,
	class OffsetIterator,
    class OutputIterator,
	class BinaryFunction = ::rocprim::minus<typename std::iterator_traits<OutputIterator>::value_type>
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void segmented_transform_kernel(
						InputIterator inputs,
						OffsetIterator begin_offsets,
						OffsetIterator end_offsets,
						OutputIterator outputs,
						BinaryFunction transform_op = BinaryFunction())
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;

	static_assert(std::is_convertible<input_type, output_type>::value,
	                  "The type OutputIterator must be such that an object of type OffsetIterator "
	                  "can be dereferenced and then implicitly converted to OutputIterator.");

	constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto segment_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();

	const unsigned int begin_offset = begin_offsets[segment_id];
	const unsigned int end_offset = end_offsets[segment_id];

	// Empty segment
	if(end_offset <= begin_offset)
	{
	    return;
	}

	const output_type segment_value = static_cast<output_type>(inputs[segment_id]);
    output_type values[ItemsPerThread];
	
	unsigned int block_offset = begin_offset;

	// Load next full blocks and continue transforming
	for(;block_offset + items_per_block <= end_offset; block_offset += items_per_block)
	{
		::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values
        );
		
		#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
	        values[i] = transform_op(values[i], segment_value);
        }

        ::rocprim::block_store_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values
        );
	}
		
	if(block_offset < end_offset)
	{
		const unsigned int valid_count = end_offset - block_offset;
		::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values,
            valid_count
        );

        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(BlockSize * i + flat_id < valid_count)
            {
                values[i] = transform_op(values[i], segment_value);
            }
        }

        ::rocprim::block_store_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values,
            valid_count
        );
	} 
}

void count_sending_spikes_temporary_storage_size(const unsigned int sending_count,
														const unsigned int segments,
														unsigned int* block_rowptrs,
														unsigned int* active_rowptrs,
														size_t& storage_size_bytes,
														hipStream_t stream)
{
	constexpr auto items_per_block = HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD;
	const unsigned int number_of_blocks =
			std::max(1u, divide_up<unsigned int>(sending_count, items_per_block));
	size_t temporary_storage_size_bytes = 0;
	
	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                temporary_storage_size_bytes,
                                active_rowptrs + segments + 1,
                                active_rowptrs,
                                0,
                                segments + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));
	storage_size_bytes = std::max(temporary_storage_size_bytes, sending_count * sizeof(unsigned int));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                temporary_storage_size_bytes,
                                block_rowptrs + number_of_blocks + 1,
                                block_rowptrs,
                                0,
                                number_of_blocks + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	storage_size_bytes = std::max(temporary_storage_size_bytes, storage_size_bytes);
}

void update_sending_spikes_gpu(const unsigned char* f_actives,
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

	constexpr auto items_per_block = HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD;
	const unsigned int number_of_blocks =
			std::max(1u, divide_up<unsigned int>(sending_count, items_per_block));
	
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(expand_and_reduce_kernel<HIP_THREADS_PER_BLOCK, unsigned int, HIP_ITEMS_PER_THREAD>),
					dim3(number_of_blocks), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					f_actives,
					sending_colinds,
					sending_count,
					f_sending_actives,
					block_rowptrs + number_of_blocks + 1);
	HIP_POST_KERNEL_CHECK("expand_and_reduce_kernel");

	size_t  storage_size_bytes = 4; 
	
    HIP_CHECK(
        rocprim::segmented_reduce(
            static_cast<void*>(active_colinds),
            storage_size_bytes,
            f_sending_actives,
            active_rowptrs + segments + 1,
            segments,
            sending_rowptrs,
            sending_rowptrs + 1,
            ::rocprim::plus<unsigned int>(), 
            0,
            stream
        )
    );
	
	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                storage_size_bytes,
                                active_rowptrs + segments + 1,
                                active_rowptrs,
                                0,
                                segments + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(static_cast<void*>(active_colinds),
                                storage_size_bytes,
                                active_rowptrs + segments + 1,
                                active_rowptrs,
                                0,
                                segments + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                storage_size_bytes,
                                block_rowptrs + number_of_blocks + 1,
                                block_rowptrs,
                                0,
                                number_of_blocks + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(static_cast<void*>(active_colinds),
                                storage_size_bytes,
                                block_rowptrs + number_of_blocks + 1,
                                block_rowptrs,
                                0,
                                number_of_blocks + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(exclusive_scan_and_select_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
					dim3(number_of_blocks), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					f_sending_actives,
					sending_count,
					block_rowptrs,
					active_colinds,
					0);
	HIP_POST_KERNEL_CHECK("exclusive_scan_and_select_kernel");

	if(segments > 1)
	{
		hipLaunchKernelGGL(
						HIP_KERNEL_NAME(segmented_transform_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
						dim3(segments - 1), 
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						sending_rowptrs + 1,
						active_rowptrs + 1,
						active_rowptrs + 2,
						active_colinds);
		HIP_POST_KERNEL_CHECK("segmented_transform_kernel");
		
	}
	
}

void update_recving_spikes_gpu(const unsigned int* inputs,
									const unsigned int* rowptrs,
									const unsigned int  segments,
									unsigned int* outputs,	
									hipStream_t stream)
{
	if(segments > 1)
	{
		hipLaunchKernelGGL(
						HIP_KERNEL_NAME(segmented_transform_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
						dim3(segments - 1), 
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						inputs + 1,
						rowptrs + 1,
						rowptrs + 2,
						outputs,
						::rocprim::plus<unsigned int>());
		HIP_POST_KERNEL_CHECK("segmented_transform_kernel");
	}
}

void update_routing_spikes_gpu(const unsigned int* inputs,
									const size_t size,
									unsigned char* outputs,
									hipStream_t stream)
{
	constexpr auto items_per_block = HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD;
	const unsigned int number_of_blocks =
			std::max(1u, divide_up<unsigned int>(size, items_per_block));
	hipLaunchKernelGGL(
						HIP_KERNEL_NAME(fill_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
						dim3(number_of_blocks), 
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						inputs,
						size,
						outputs,
						1);
		HIP_POST_KERNEL_CHECK("fill_kernel");
}

void update_routing_neuron_ids_gpu(const unsigned int* inputs,
											const size_t size,
											unsigned int* outputs,	
											hipStream_t stream)
{
	constexpr auto items_per_block = HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD;
	const unsigned int number_of_blocks =
			std::max(1u, divide_up<unsigned int>(size, items_per_block));
	hipLaunchKernelGGL(
						HIP_KERNEL_NAME(transform_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
						dim3(number_of_blocks), 
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						inputs,
						size,
						outputs);
		HIP_POST_KERNEL_CHECK("transform_kernel");
}

template void init_spike_time_gpu<float>(const unsigned int n,
										const float val,
										float* t_actives,
										hipStream_t stream);

}//namespace istbi 
