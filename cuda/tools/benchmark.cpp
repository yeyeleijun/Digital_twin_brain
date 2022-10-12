#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <future>
#include <chrono>
#include <functional>
#include <thread>
#include <string>
#include <cassert>
#include <cstring>
#include <set>
#include <unordered_set>
#include <stdint.h>
#include <mpi.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include "common.hpp"
#include "block.hpp"
#include "blocking_queue.hpp"
#include "device_function.hpp"
#include "util/transpose.hpp"
#include "util/cmd_arg.hpp"
#include <json/json.h>
#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>
#include "snn.grpc.pb.h"
#include "logging.hpp"
#include "notification.hpp"
#include "unique.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define ENV_LOCAL_RANK 		"OMPI_COMM_WORLD_LOCAL_RANK"
#define MPI_MASTER_RANK		0

#define ID2RANK(id) (static_cast<int>(id) + 1)
#define RANK2ID(rank) static_cast<unsigned short>((rank) - 1)

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   					\
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static constexpr std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds::zero();

static constexpr auto INVALID_ACTIVE_OFFSET = -1;
using namespace std;
using namespace dtb;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::ServerReaderWriter;
using grpc::Status;
using snn::SnnStatus;
using snn::SubblockInfo;
using snn::InitRequest;
using snn::InitResponse;
using snn::RunRequest;
using snn::RunResponse;
using snn::MetricRequest;
using snn::MetricInfo;
using snn::MetricResponse;
using snn::PropType;
using snn::UpdatePropRequest;
using snn::UpdatePropResponse;
using snn::UpdateGammaRequest;
using snn::UpdateGammaResponse;
using snn::UpdateGammaWithResultResponse;
using snn::UpdateHyperParaRequest;
using snn::UpdateHyperParaResponse;
using snn::UpdateSampleRequest;
using snn::UpdateSampleResponse;
using snn::ShutdownRequest;
using snn::ShutdownResponse;
using snn::Snn;
using chrono::steady_clock;
using chrono::time_point;
using chrono::duration;

enum Command
{
	SNN_INIT,
	SNN_RUN,
	SNN_MEASURE,
	SNN_UPDATE_PROP,
	SNN_UPDATE_GAMMA,
	SNN_UPDATE_HYPERPARA,
	SNN_SAMPLE,
	SNN_SHUTDOWN
};

enum Tag
{
	TAG_UPDATE_PROP = 1000,
	TAG_UPDATE_GAMMA = 3000,
	TAG_UPDATE_HYPERPARA = 4000,
	TAG_UPDATE_SAMPLE = 5000,
	TAG_REPORT_NAME = 6000
};

template<typename T>
struct RunReportInfo
{
	unique_ptr<DataAllocator<unsigned int>> freqs_;
	unique_ptr<DataAllocator<T>> vmeans_;
	unique_ptr<DataAllocator<char>> spikes_;
	unique_ptr<DataAllocator<T>> vmembs_;
	unique_ptr<DataAllocator<T>> imeans_;
};

template<typename T>
struct MetricData
{
	T max_duration_;
	T min_duration_;
	T avg_duration_;
};

struct TransTable
{
	enum Action
	{
		ACTION_RECORD,
		ACTION_MERGE,
		ACTION_ROUTE
	};

	enum Mode
	{
		MODE_POINT_TO_POINT,
		MODE_ROUTE_WITHOUT_MERGE,
		MODE_ROUTE_WITH_MERGE
	};

	enum MemoryType
	{
		MEMORY_CPU,
		MEMORY_GPU
	};

	enum PackageType
	{
		PACKAGE_SINGLE,
		PACKAGE_MERGE
	};

	enum SendStrategy
	{
		STRATEGY_SEND_SEQUENTIAL,
		STRATEGY_SEND_PAIRWISE,
		STRATEGY_SEND_RANDOM
	};

	//Storing pointer of  buffer in first element, the number to be sent in 2nd. 
	using BufferRecord = tuple<unsigned int*, int>;
	using BufferMetaData = tuple<unique_ptr<DataAllocator<unsigned int>>, int>;
	
	struct Config
	{
		using RouteInfo = map<int, BufferRecord>;
		void clear()
		{
			sending_ranks_.clear();
			routing_ranks_.clear();
			recving_ranks_.clear();
			routing_infos_.clear();
		}

		//Storing ranks to be sent by local rank.
		unordered_set<int> sending_ranks_;
		//Storing level of routing hierarchy in first element of tuple structure, and ranks to be routed in 2nd.
		unordered_map<int, tuple<int, set<int>, bool>> routing_ranks_;
		//Storing ranks to be received by local rank.
		map<int, int> recving_ranks_;
		//key of the outer unordered map indicates target rank, and key in the inner unordered map indicates rank of receiver.
		//Storing neuron offset of brain block in first element in the tuple structure, pointer of buffer in 2rd,
		//and number of buffer.
		unordered_map<int, RouteInfo> routing_infos_;
	};

	struct SendMetaData
	{
		//rank of receiver
		int receiver_rank_;
		//buffer information.
		BufferRecord buffer_;
	};

	struct RecvMetaData
	{
		//rank of sender
		int sender_rank_;
		int sender_tag_;
		//indicating action type, only including ACTION_RECORD, ACTION_MERGE, ACTION_ROUTE.
		Action action_;
		//buffer information.
		BufferMetaData buffer_;
	};

	struct RecvRecord
	{
		struct MergeRecord
		{
			//recording ranks of all receipt.
			set<int> receipt_set_;
			//indicating action type, only including ACTION_RECORD, ACTION_ROUTE.
			vector<Action> receipt_actions_;
		};
		
		//If it is forwarded, the data resides in the CPU memory; 
		//if it is local, the data resides in the GPU memory.
		MemoryType type_;
		//buffer information recording.
		BufferRecord record_;

		static unordered_map<int, MergeRecord> merge_records_;
	};

	struct RouteRecord
	{
		//ranks to be routed
		set<int> receiver_ranks_;
		
		//routing information
		unique_ptr<DataAllocator<unsigned int>> receiver_rowptrs_;
		unique_ptr<DataAllocator<unsigned int>> receiver_colinds_;
		unique_ptr<DataAllocator<unsigned char>> f_actives_;
		
		unique_ptr<DataAllocator<unsigned int>> receiver_block_rowptrs_;
		unique_ptr<DataAllocator<unsigned int>> receiver_active_rowptrs_;
		unique_ptr<DataAllocator<unsigned int>>	receiver_active_colinds_;
		unique_ptr<DataAllocator<unsigned char>> f_receiver_actives_;

		//peer rank from which the local rank receives.
		int sender_rank_;
		//the number of unsigned integers received from sender_rank_.
		unsigned int sender_count_;
		unique_ptr<DataAllocator<unsigned int>> sender_active_colinds_;

		struct MergeRecord
		{
			//index of route_buffs_ .
			int index_;
			//indicating action type, only including ACTION_ROUTE,  ACTION_MERGE.
			Action action_;
			//recording ranks of all receipt for routing merge.
			unordered_set<int> receipt_set_;
		};

		static unordered_map<unsigned int, shared_ptr<MergeRecord>> merge_records_;
	};

	struct RouteMetaData
	{
		//offset of merging segment.
		int offset_;

		//rank of sender
		int receiver_rank_;
		int receiver_tag_;
		
		//ranks receiving from peer rank.
		unordered_set<int> receipt_set_;
		//buffer information.
		BufferMetaData buffer_;

		vector<tuple<int, unsigned int*, int, MemoryType>> buff_records_;
	};

	Mode mode_;

	vector<SendMetaData> send_buffs_;

	//key indicates target rank of sender and value indicates meta data of operation to each sender
	//in which key indicates  rank of sender.
	unordered_map<int, RecvRecord> recv_table_;
	vector<RecvMetaData> recv_buffs_;
	
	unordered_map<int, RouteRecord> route_table_;
	//for convience, requests to be merged are put in the front part of route_buffs_ , and requests
	//to be forward are put in the latter part. Hence, routing_merged_infos_ correspond to the front part 
	//of routing_infos_ one to one.
	vector<RouteMetaData> route_buffs_;
	
	//communication between blocks
	vector<MPI_Request> send_requests_;
	vector<MPI_Request> recv_requests_;
	vector<MPI_Request> route_requests_;

	//Knuth-Durstenfeld Shuffle
	void random_shuffle(vector<int>& indice)
	{
		int cur_idx;
		srand((unsigned)time(NULL));
		for(int i = (int)indice.size() - 1; i >= 1; i--)
		{
			cur_idx = rand() % (i + 1);
			if(cur_idx != i)
			{
				std::swap(indice[cur_idx], indice[i]);
			}
		}
	}

	void pairwise(const int rank, vector<int>& indice)
	{
		indice.reserve(send_buffs_.size());
		for(const auto& meta : send_buffs_)
		{
			indice.push_back(meta.receiver_rank_);
		}

		thrust::device_vector<int> d_ranks = indice;
		assert(thrust::is_sorted(d_ranks.begin(), d_ranks.end()));
		auto it = thrust::upper_bound(d_ranks.begin(), d_ranks.end(), rank);
		size_t idx = it - d_ranks.begin();
		for(size_t i = 0; i < indice.size(); i++)
		{
			indice[i] = static_cast<int>((idx++) % indice.size());
		}
	}

	void clear()
	{
		mode_ = Mode::MODE_POINT_TO_POINT;
		send_buffs_.clear();
		recv_table_.clear();
		recv_buffs_.clear();
		route_table_.clear();
		route_buffs_.clear();
		send_requests_.clear();
		recv_requests_.clear();
		route_requests_.clear();
		RecvRecord::merge_records_.clear();
		RouteRecord::merge_records_.clear();
	}
};

unordered_map<int, TransTable::RecvRecord::MergeRecord> TransTable::RecvRecord::merge_records_;
unordered_map<unsigned int, shared_ptr<TransTable::RouteRecord::MergeRecord>> TransTable::RouteRecord::merge_records_;

struct MPIInfo
{
	MPIInfo(const int rank, const int size, const int tag, MPI_Comm comm)
	:rank_(rank),
	size_(size),
	comm_(comm){}
	
	int rank_;
	int size_;
	MPI_Comm comm_;
};
	
template<typename T, typename T2>
class NodeInfo
{
public:
	
	NodeInfo(const int rank, const int size, const int tag, const MPI_Comm comm, 
			const int gpu_id, const string& name, int timeout,
			const set<int>& rank_in_same_node)
	:info_(new MPIInfo(rank, size, tag, comm)),
	gid_(gpu_id),
	timeout_(timeout),
	rank_in_same_node_(rank_in_same_node)
	{
		name_ = name + string("-") + std::to_string(rank);
		trans_table_ = make_unique<TransTable>();
		trans_table_->mode_ = TransTable::Mode::MODE_POINT_TO_POINT;
	}

	void clear()
	{
		block_.reset(nullptr);
		samples_.reset(nullptr);
		spikes_.reset(nullptr);
		vmembs_.reset(nullptr);
		trans_table_->clear();
		recving_queue_.clear();
		routing_queue_.clear();
		reporting_queue_.clear();

		computing_duration_.clear();
		reporting_duration_.clear();
		duration_inter_node_.clear();
		duration_intra_node_.clear();
		
		sending_byte_size_inter_node_.clear();
		sending_byte_size_intra_node_.clear();
		recving_byte_size_inter_node_.clear();
		recving_byte_size_intra_node_.clear();
	}

	int gid_;
	string name_;
	unique_ptr<MPIInfo> info_;
	set<int> rank_in_same_node_;
	unique_ptr<BrainBlock<T, T2>> block_;

	unique_ptr<TransTable> trans_table_;
	
	//per node
	unique_ptr<DataAllocator<unsigned int>> samples_;
	unique_ptr<DataAllocator<char>> spikes_;
	unique_ptr<DataAllocator<T>> vmembs_;

	Command cmd_;
	string path_;
	//T noise_rate_;
	T delta_t_;

	int send_strategy_;
	int iter_;
	int iter_offset_;
	bool has_freq_;
	bool has_vmean_;
	bool has_sample_;
	bool has_imean_;
	int timeout_;

	//gamma
	vector<unsigned int> prop_indice_;
	vector<unsigned int> brain_indice_;
	bool has_prop_;
	
	Notification reporting_notification_;
	
	BlockingQueue<int> recving_queue_;
	BlockingQueue<int> routing_queue_;
	BlockingQueue<shared_ptr<RunReportInfo<T>>> reporting_queue_;

	double used_cpu_mem;
	double total_gpu_mem;
	double used_gpu_mem;

	vector<double> computing_duration_;
	vector<double> routing_duration_;
	vector<double> reporting_duration_;
	vector<double> duration_inter_node_;
	vector<double> duration_intra_node_;

	vector<uint64_t> sending_byte_size_inter_node_;
	vector<uint64_t> sending_byte_size_intra_node_;
	vector<uint64_t> recving_byte_size_inter_node_;
	vector<uint64_t> recving_byte_size_intra_node_;

	vector<double> flops_update_v_membrane_;
	vector<double> flops_update_j_presynaptic_;
	vector<double> flops_update_i_synaptic_;
};

static void init_mpi_env(int* argc, char*** argv, int& rank, int& gpu_id, int& size, string& name)
{
	// Setting the device here will have an effect only for the CUDA-aware MPI
	char* local_rank_str = NULL;
	
	// We extract the local rank initialization using an environment variable
	if((local_rank_str = getenv(ENV_LOCAL_RANK)) != NULL)
	{
		rank = atoi(local_rank_str);
	}

	int provided;
	MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);

	if(provided != MPI_THREAD_MULTIPLE)  
	{  
	    cerr << "MPI do not Support Multiple thread" << endl;  
	    exit(0);  
	} 
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(MPI_MASTER_RANK == rank)
	{
		cout << "Multi thread provide " << provided << " support." << endl;
		gpu_id = -1;
	}
	else
	{
		int dev_count = 0;
		HIP_CHECK(hipGetDeviceCount(&dev_count));
		gpu_id = (rank % dev_count);
		HIP_CHECK(hipSetDevice(gpu_id));
		hipDeviceProp_t deviceProp;
        hipGetDeviceProperties(&deviceProp, gpu_id);
		name = string(deviceProp.name);
	}
}

static int wait_handle(Command& cmd, const MPIInfo& info)
{
	return MPI_Bcast(&cmd, 1, MPI_INT, MPI_MASTER_RANK, info.comm_);
}

static int snn_sync(const MPIInfo& info)
{
	return MPI_Barrier(info.comm_);
}

static int snn_gather(const MPIInfo& info,
						const void *sendbuf,
						int sendcount,
						MPI_Datatype sendtype,
						void *recvbuf,
						int recvcount,
						MPI_Datatype recvtype,
						int root = MPI_MASTER_RANK)
{
	return MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, info.comm_);
}

static int snn_gatherv(const MPIInfo& info,
						const void *sendbuf,
						int sendcount,
						MPI_Datatype sendtype,
						void *recvbuf,
						const int* recvcounts,
						const int* displs,
						MPI_Datatype recvtype,
						int root = MPI_MASTER_RANK)
{
	return MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
			displs, recvtype, root, info.comm_);
}

static void snn_init_report(const MPIInfo& info,
							int* recvcounts,
							int* displs,
							int* neurons_per_block,
							vector<int>& subids,
							vector<int>& subcounts,
							double* used_cpu_mems,
							double* total_gpu_mems,
							double* used_gpu_mems)
{
	
	int i_data = 0;
	double data = 0.;
	int err;

	err = snn_gather(info, &i_data, 1, MPI_INT, neurons_per_block, 1, MPI_INT);
	assert(err == MPI_SUCCESS);
	
	err = snn_gather(info, &i_data, 1, MPI_INT, recvcounts, 1, MPI_INT);
	assert(err == MPI_SUCCESS);
	
	int total = 0;
	for(int i = 0; i < info.size_; i++)
	{
		total += recvcounts[i];
	}
	subids.resize(total);
	subcounts.resize(total);
	
	thrust::exclusive_scan(recvcounts, recvcounts + info.size_, displs);

	err = snn_gatherv(info, NULL, i_data, MPI_INT, subids.data(), recvcounts, displs, MPI_INT);
	assert(err == MPI_SUCCESS);

	err = snn_gatherv(info, NULL, i_data, MPI_INT, subcounts.data(), recvcounts, displs, MPI_INT);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &data, 1, MPI_DOUBLE, used_cpu_mems, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);
	
	err = snn_gather(info, &data, 1, MPI_DOUBLE, total_gpu_mems, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);
	
	err = snn_gather(info, &data, 1, MPI_DOUBLE, used_gpu_mems, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

}

template<typename T, typename T2>
static void snn_init_report(NodeInfo<T, T2>& node)
{
	int neurons = static_cast<int>(node.block_->get_total_neurons());
	int subblocks = static_cast<int>(node.block_->get_total_subblocks());
	int err;
	
	err = snn_gather(*node.info_, &neurons, 1, MPI_INT, NULL, 1, MPI_INT);
	assert(err == MPI_SUCCESS);
	
	err = snn_gather(*node.info_, &subblocks, 1, MPI_INT, NULL, 1, MPI_INT);
	assert(err == MPI_SUCCESS);

	err = snn_gatherv(*node.info_, node.block_->get_sub_bids_cpu(),
					subblocks, MPI_UNSIGNED, NULL, NULL, NULL, MPI_INT);
	assert(err == MPI_SUCCESS);

	err = snn_gatherv(*node.info_, node.block_->get_sub_bcounts_cpu(),
					subblocks, MPI_UNSIGNED, NULL, NULL, NULL, MPI_INT);
	assert(err == MPI_SUCCESS);

	err = snn_gather(*node.info_, &node.used_cpu_mem, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(*node.info_, &node.total_gpu_mem, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(*node.info_, &node.used_gpu_mem, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

}

static void snn_init(const MPIInfo& info,
					string& path,
					vector<char>& route_json,
					float& delta_t,
					int& comm_mode)
{
	size_t len, rlen = 0;
	vector<char> vpath;
	int err;
	
	if(MPI_MASTER_RANK == info.rank_)
	{
		len = path.length();
		rlen = route_json.size();
	}
	
	err = MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, MPI_MASTER_RANK, info.comm_);
	assert(err == MPI_SUCCESS);
	if(len > 0)
	{
		vpath.resize(len + 1);
		vpath[len] = 0;

		if(MPI_MASTER_RANK == info.rank_)
		{
			path.copy(vpath.data(), len);
		}
	
		err = MPI_Bcast(vpath.data(), len, MPI_CHAR, MPI_MASTER_RANK, info.comm_);
		assert(err == MPI_SUCCESS);

		if(MPI_MASTER_RANK != info.rank_)
		{
			path.assign(vpath.data());
		}
	}

	err = MPI_Bcast(&rlen, 1, MPI_UNSIGNED_LONG, MPI_MASTER_RANK, info.comm_);
	assert(err == MPI_SUCCESS);
	
	if(rlen > 0)
	{
		if(MPI_MASTER_RANK != info.rank_)
		{
			route_json.resize(rlen);
		}

		const size_t MAX_SIZE = 1 * 1000 * 1000 * 1000;
		int bytes;
		size_t offset = 0;
		do{
			bytes = (rlen > MAX_SIZE) ? (int)MAX_SIZE : (int)rlen;
			if(MPI_MASTER_RANK == info.rank_)
			{
				std::cout << "Route json bytes: " << (offset + bytes) << std::endl;
			}
			err = MPI_Bcast(route_json.data() + offset, bytes, MPI_CHAR, MPI_MASTER_RANK, info.comm_);
			assert(err == MPI_SUCCESS);
			rlen -= (size_t)bytes;
			offset += (size_t)bytes;
		}while(rlen > 0);
	}

	assert(rlen == 0);
	
	err = MPI_Bcast(&delta_t, 1, MPI_FLOAT, MPI_MASTER_RANK, info.comm_);
	assert(err == MPI_SUCCESS);

	err = MPI_Bcast(&comm_mode, 1, MPI_INT, MPI_MASTER_RANK, info.comm_);
	assert(err == MPI_SUCCESS);
}


template<typename T, typename T2>
static void config_conn_table(const std::string& filename,
								NodeInfo<T, T2>& node)

{
	node.block_ = make_unique<BrainBlock<T, T2>>(RANK2ID(node.info_->rank_), node.gid_, node.delta_t_);
	node.block_->init_connection_table_gpu(filename);
	node.block_->init_config_params_gpu(filename);
	node.block_->init_all_stages_gpu();
	node.block_->reset_V_membrane_gpu();
	HIP_CHECK(hipDeviceSynchronize());
}

static bool config_trans_table(vector<char>& route_json,
								const MPIInfo& info,
								TransTable::Config& conf)
{
	bool ret = true;
	if(!route_json.empty())
	{
		std::stringbuf str_buf;
		str_buf.pubsetbuf(route_json.data(), route_json.size());
		std::istream is(&str_buf);
		Json::CharReaderBuilder builder;
		builder["collectComments"] = false;
		Json::Value root;
		JSONCPP_STRING errs;

		if (Json::parseFromStream(builder, is, &root, &errs))
		{
			auto members = root.getMemberNames();
			for (Json::ArrayIndex i = 0; i < members.size(); ++i)
			{
				char *end;
				int dst_rank = ID2RANK(std::strtol(members[i].c_str(), &end, 10));
				const Json::Value obj = root[members[i]];
				const Json::Value src = obj["src"];
				const Json::Value dst = obj["dst"];
				assert(src.size() == dst.size());

				thrust::host_vector<int> h_sranks(src.size());
				thrust::host_vector<int> h_dranks(dst.size());
				thrust::device_vector<int> d_sranks(src.size());
				thrust::device_vector<int> d_dranks(dst.size());
				
				for(Json::ArrayIndex j = 0; j < src.size(); ++j)
				{
					h_sranks[j] = ID2RANK(src[j].asInt());
					h_dranks[j] = ID2RANK(dst[j].asInt());
				}

				HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_sranks.data()), h_sranks.data(), h_sranks.size() * sizeof(int), hipMemcpyHostToDevice));
				HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_dranks.data()), h_dranks.data(), h_dranks.size() * sizeof(int), hipMemcpyHostToDevice));
				
				auto it = thrust::unique(d_dranks.begin(), d_dranks.end());
				if(it != d_dranks.end())
				{
					std::cerr << "Rank" << info.rank_ << " config route table failed. fall back peer to peer" << std::endl;
					conf.clear();
					ret = false;
					break;
				}

				if(dst_rank != info.rank_)
				{
					it = thrust::find(d_dranks.begin(), d_dranks.end(), info.rank_);
					int next_rank = -1;
					if(it != d_dranks.end())
					{
						assert(1 == thrust::count(d_dranks.begin(), d_dranks.end(), info.rank_));
						next_rank = h_sranks[it - d_dranks.begin()];
						assert(conf.recving_ranks_.emplace(std::make_pair(dst_rank, next_rank)).second);
					}
					
					if(!thrust::is_sorted(d_sranks.begin(), d_sranks.end()))
					{
						thrust::sort_by_key(d_sranks.begin(), d_sranks.end(), d_dranks.begin());
					}

					int count = thrust::count(d_sranks.begin(), d_sranks.end(), info.rank_);
					if(count > 0)
					{
						std::set<int> paths;
						if(next_rank < 0)
						{
							std::cerr << "Rank" << info.rank_ << " config route table failed. fall back peer to peer" << std::endl;
							conf.clear();
							ret = false;
							break;
						}
						
						assert(paths.insert(next_rank).second);
						it = thrust::lower_bound(d_sranks.begin(), d_sranks.end(), info.rank_);
						assert(it != d_sranks.end());
						int offset = it - d_sranks.begin();
						HIP_CHECK(hipMemcpy(h_dranks.data(), thrust::raw_pointer_cast(d_dranks.data() + offset), count * sizeof(int), hipMemcpyDeviceToHost));
						for(int j = 0; j < count; j++)
						{
							assert(h_dranks[j] != info.rank_);
						}
						auto route = conf.routing_ranks_.emplace(std::make_pair(dst_rank, tuple<int, set<int>, bool>()));
						assert(route.second);
						
						for(int j = 0; j < count; j++)
						{
							 assert(std::get<1>(route.first->second).insert(h_dranks[j]).second);
						}
						
						if(next_rank != dst_rank)
						{
							HIP_CHECK(hipMemcpy(h_sranks.data(), thrust::raw_pointer_cast(d_sranks.data()), d_sranks.size() * sizeof(int), hipMemcpyDeviceToHost));
							do
							{
								it = thrust::find(d_dranks.begin(), d_dranks.end(), next_rank);
								assert(it != d_dranks.end());
								next_rank = h_sranks[it - d_dranks.begin()];
								assert(paths.insert(next_rank).second);
							}while(next_rank != dst_rank);
						}

						std::get<0>(route.first->second) = static_cast<int>(paths.size());
						std::get<2>(route.first->second) = false;
					}
				}
				else
				{
					if(!thrust::is_sorted(d_sranks.begin(), d_sranks.end()))
					{
						thrust::sort_by_key(d_sranks.begin(), d_sranks.end(), d_dranks.begin());
					}

					int count = thrust::count(d_sranks.begin(), d_sranks.end(), info.rank_);
					if(count > 0)
					{
						it = thrust::lower_bound(d_sranks.begin(), d_sranks.end(), info.rank_);
						if(it != d_sranks.end())
						{
							int offset = it - d_sranks.begin();
							HIP_CHECK(hipMemcpy(h_dranks.data(), thrust::raw_pointer_cast(d_dranks.data() + offset), count * sizeof(int), hipMemcpyDeviceToHost));
							for(unsigned int j = 0; j < count; j ++)
							{
								assert(conf.sending_ranks_.insert(h_dranks[j]).second);
							}
						}
					}
					else
					{
						std::cerr << "Rank" << info.rank_ << " config route failed. fall back peer to peer" << std::endl;
						conf.clear();
						ret = false;
						break;
					}
				}
			}

		}
	}
	else
	{
		std::cerr << "Rank" << info.rank_ << " route configuration file not exists. fall back peer to peer" << std::endl;
		ret = false;
	}

	if(!ret)
	{
		for(int rank = 0; rank < info.size_; rank++)
		{
			if(MPI_MASTER_RANK == rank || rank == info.rank_)
				continue;

			assert(conf.recving_ranks_.emplace(std::make_pair(rank, rank)).second);
			assert(conf.sending_ranks_.insert(rank).second);
		}
	}

	return ret;

}

static std::ostream& operator<<(std::ostream& out, const TransTable::Config& conf)
{
	out << "recv table: " << std::endl;
	for(const auto& p : conf.recving_ranks_)
	{
		out << "[" << p.first << " : " << p.second << "] ";
	}
	out << std::endl;
	
	out << "route table: " << std::endl;
	for(const auto& p : conf.routing_ranks_)
	{
		out << "[" << p.first << " : ";
		for(const auto& q : std::get<1>(p.second))
		{
			out << q << " ";
		}
		out << "]" << std::endl;
	}

	out << "send table: " << std::endl;
	out << "[";
	for(const auto& p : conf.sending_ranks_)
	{
		out << p << " ";
	}
	out << "]" << std::endl;

	return out;
}

//
template<typename T, typename T2>
static void send_init(const NodeInfo<T, T2>& node,
					TransTable::Config& conf,
					vector<MPI_Request>& requests,
					vector<TransTable::BufferRecord>& sending_buffs)
{
	const unsigned short* receiver_bids = node.block_->f_receiving_bids_.data();
	const unsigned int* receiver_rowptrs = node.block_->f_receiving_rowptrs_->cpu_data();
	const unsigned int* receiver_colinds = node.block_->f_receiving_colinds_->cpu_data();
	unsigned int last_index = 0;

	for (auto recv_it = conf.recving_ranks_.begin(); recv_it != conf.recving_ranks_.end();)
	{
		int count = 0;
		if(last_index < node.block_->f_receiving_bids_.size() && 
			recv_it->first == ID2RANK(receiver_bids[last_index]))
		{
			count = static_cast<int>(receiver_rowptrs[last_index + 1] - receiver_rowptrs[last_index]);
			assert(count > 0);
		}

		auto route_it = conf.routing_ranks_.find(recv_it->first);
		//Point-to-point traffic or no routing is required
		if(conf.routing_ranks_.empty() || route_it == conf.routing_ranks_.end())
		{
			sending_buffs.push_back(TransTable::BufferRecord());
			if(count > 0)
			{
				std::get<0>(sending_buffs.back()) = const_cast<unsigned int*>(receiver_colinds) + receiver_rowptrs[last_index];
				last_index++; 
			}
			else
			{
				std::get<0>(sending_buffs.back()) = nullptr;
			}

			std::get<1>(sending_buffs.back()) = count;

			MPI_Request request;
			int err = MPI_Isend(std::get<0>(sending_buffs.back()),
								std::get<1>(sending_buffs.back()),
								MPI_UNSIGNED,
								recv_it->second,
								recv_it->first,
								node.info_->comm_,
								&request);
			assert(err == MPI_SUCCESS);
			requests.push_back(request);

			if(0 == count)
			{
				recv_it = conf.recving_ranks_.erase(recv_it);
			}
			else
			{
				recv_it++;
			}
		}
		else
		{
			if(count > 0)
			{
				assert(std::get<1>(route_it->second).insert(node.info_->rank_).second);
				//route merging may be required locally
				std::get<2>(route_it->second) = true;
				auto route = conf.routing_infos_.emplace(std::make_pair(recv_it->first, TransTable::Config::RouteInfo()));
				assert(route.second);
				auto info = route.first->second.emplace(std::make_pair(node.info_->rank_, TransTable::BufferRecord()));
				assert(info.second);
				std::get<0>(info.first->second) = const_cast<unsigned int*>(receiver_colinds) + receiver_rowptrs[last_index];
				std::get<1>(info.first->second) = count;
				last_index++;
			}
			recv_it++;
		}
	}	
	assert(last_index == node.block_->f_receiving_bids_.size());
}

template<typename T, typename T2>
static void union_route(NodeInfo<T, T2>& node,
						const TransTable::Config::RouteInfo& route_info,
						TransTable::RouteRecord& record)
{
	if(1 >= route_info.size())
		return;

	vector<int> ranks;
	vector<unsigned int> rowptrs;
	vector<unsigned int*> colinds;
	thrust::device_vector<unsigned int> d_unions;
	unsigned int union_size = 0;
	
	ranks.reserve(route_info.size());
	rowptrs.reserve(route_info.size() + 1);
	colinds.reserve(route_info.size());

	unsigned int count = 0;
	rowptrs.push_back(count);
	for(auto pair : route_info)
	{
		ranks.push_back(pair.first);
		count += static_cast<unsigned int>(std::get<1>(pair.second));
		rowptrs.push_back(count);
		colinds.push_back(std::get<0>(pair.second));
		
		if(0 == union_size)
		{
			d_unions.resize(count);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_unions.data()), std::get<0>(pair.second), count * sizeof(unsigned int), hipMemcpyHostToDevice));
			union_size += count;
		}
		else
		{
			thrust::device_vector<unsigned int> d_data1(union_size);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_data1.data()), thrust::raw_pointer_cast(d_unions.data()), union_size * sizeof(unsigned int), hipMemcpyDeviceToDevice));
			thrust::device_vector<unsigned int> d_data2(std::get<1>(pair.second));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_data2.data()), std::get<0>(pair.second), d_data2.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
			union_size += static_cast<unsigned int>(std::get<1>(pair.second));
			if(d_unions.size() < union_size)
			{
				d_unions.resize(union_size);
			}
			thrust::device_vector<unsigned int>::iterator it = thrust::set_union(d_data1.begin(), d_data1.end(), d_data2.begin(), d_data2.end(), d_unions.begin());
			union_size = it - d_unions.begin();
		}
	}
	
	assert(union_size > 0);
	
	record.receiver_rowptrs_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, rowptrs.size() * sizeof(unsigned int));
	memcpy(record.receiver_rowptrs_->mutable_cpu_data(), rowptrs.data(), record.receiver_rowptrs_->size());
	HIP_CHECK(hipMemcpy(record.receiver_rowptrs_->mutable_gpu_data(), record.receiver_rowptrs_->cpu_data(), record.receiver_rowptrs_->size(), hipMemcpyHostToDevice));
	
	record.receiver_colinds_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_,sizeof(unsigned int) * count);
	for(unsigned int idx = 0; idx < ranks.size(); idx++)
	{
		assert(record.receiver_ranks_.insert(ranks[idx]).second);
		HIP_CHECK(hipMemcpy(record.receiver_colinds_->mutable_gpu_data() + rowptrs[idx], colinds[idx], sizeof(unsigned int) * (rowptrs[idx + 1] - rowptrs[idx]), hipMemcpyHostToDevice));
	}

	record.receiver_active_rowptrs_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, 2 * rowptrs.size() * sizeof(unsigned int));
	record.receiver_active_rowptrs_->gpu_data();
	record.receiver_active_rowptrs_->cpu_data();
	
	unsigned int blocks = divide_up<unsigned int>(count, HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD);
	record.receiver_block_rowptrs_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, 2 * (blocks + 1) * sizeof(unsigned int));
	record.receiver_block_rowptrs_->gpu_data();

	size_t storage_size_bytes = 0;
	node.block_->count_F_sending_actives_temporary_storage_size(count,
												rowptrs.size() - 1,
												record.receiver_block_rowptrs_->mutable_gpu_data(),
												record.receiver_active_rowptrs_->mutable_gpu_data(),
												storage_size_bytes);
	assert(storage_size_bytes >= 4);
	
	record.receiver_active_colinds_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, storage_size_bytes);
	record.receiver_active_colinds_->gpu_data();

	record.f_receiver_actives_ = make_unique<DataAllocator<unsigned char>>(node.info_->rank_, count * sizeof(unsigned char));
	record.f_receiver_actives_->gpu_data();
	record.sender_active_colinds_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * union_size);
	HIP_CHECK(hipMemcpy(record.sender_active_colinds_->mutable_gpu_data(), thrust::raw_pointer_cast(d_unions.data()), record.sender_active_colinds_->size(), hipMemcpyDeviceToDevice));
	record.f_actives_ = make_unique<DataAllocator<unsigned char>>(node.info_->rank_, sizeof(unsigned char) * union_size);
	record.f_actives_->gpu_data();
}

template<typename T, typename T2>
static void route_init(NodeInfo<T, T2>& node,
					TransTable::Config& conf,
					const int rank,
					const int tag,
					const unsigned int buff_index,
					vector<vector<unsigned int>>& recving_buffs,
					vector<MPI_Request>& requests,
					vector<TransTable::BufferRecord>& sending_buffs,
					map<int, set<unsigned int>>& route_indice)
{
	MPI_Request request;
	auto route_it = conf.routing_ranks_.find(tag);
	if(route_it == conf.routing_ranks_.end())
	{
		LOG_INFO << "[route_init]: invalid routing table configuration!" << std::endl;
		assert(0);
	}

	assert(std::get<1>(route_it->second).find(rank) != std::get<1>(route_it->second).end());
	
	if(0 == recving_buffs[buff_index].size())
	{
		if(1 == std::get<1>(route_it->second).size() ||
			(2 == std::get<1>(route_it->second).size() &&
			std::get<2>(route_it->second)))
		{
			auto recv_it = conf.recving_ranks_.find(tag);
			assert(recv_it != conf.recving_ranks_.end());
			sending_buffs.push_back(TransTable::BufferRecord());
			int receiver_rank = recv_it->second;
			
			if(1 == std::get<1>(route_it->second).size())
			{
				assert(!std::get<2>(route_it->second));
			}
			
			if(std::get<2>(route_it->second))
			{	
				auto info_it = conf.routing_infos_.find(tag);
				assert(info_it != conf.routing_infos_.end());
				assert(1 == info_it->second.size() && info_it->second.begin()->first == node.info_->rank_);
				const auto& info = info_it->second[node.info_->rank_];
				assert(std::get<1>(info) > 0);
				
				std::get<0>(sending_buffs.back()) = std::get<0>(info);
				std::get<1>(sending_buffs.back()) = std::get<1>(info);
				conf.routing_infos_.erase(info_it);
				
				auto route_idx_it = route_indice.find(tag);
				if(route_idx_it != route_indice.end())
				{
					route_indice.erase(route_idx_it);
				}
			}
			else
			{
				conf.recving_ranks_.erase(recv_it);
				std::get<0>(sending_buffs.back()) = nullptr;
				std::get<1>(sending_buffs.back()) = 0;
			}

			int err = MPI_Isend(std::get<0>(sending_buffs.back()),
								std::get<1>(sending_buffs.back()),
								MPI_UNSIGNED,
								receiver_rank,
								tag,
								node.info_->comm_,
								&request);
			assert(err == MPI_SUCCESS);
			requests.push_back(request);

			conf.routing_ranks_.erase(route_it);
			return;
		}
		else
		{
			assert(1 == std::get<1>(route_it->second).erase(rank));
		}
	}

	auto info_it = conf.routing_infos_.find(tag);
	auto route_idx_it = route_indice.find(tag);

	if(0 < recving_buffs[buff_index].size())
	{
		if(info_it == conf.routing_infos_.end())
		{
			auto route = conf.routing_infos_.emplace(std::make_pair(tag, TransTable::Config::RouteInfo()));
			assert(route.second);
			info_it = route.first;
		}

		if(route_idx_it == route_indice.end())
		{
			auto route_idx = route_indice.emplace(std::make_pair(tag, set<unsigned int>()));
			assert(route_idx.second);
			route_idx_it = route_idx.first;
		}
		
		auto info = info_it->second.emplace(std::make_pair(rank, TransTable::BufferRecord()));
		assert(info.second);
		std::get<0>(info.first->second) = recving_buffs[buff_index].data();
		std::get<1>(info.first->second) = static_cast<int>(recving_buffs[buff_index].size());
		assert(route_idx_it->second.insert(buff_index).second);
	}
	
	if(info_it == conf.routing_infos_.end())
		return;
	
	if(info_it->second.size() == std::get<1>(route_it->second).size())
	{
		auto recv_it = conf.recving_ranks_.find(tag);
		assert(recv_it != conf.recving_ranks_.end());
		auto tab_it = node.trans_table_->route_table_.find(tag);
		assert(tab_it == node.trans_table_->route_table_.end());
		auto table = node.trans_table_->route_table_.emplace(std::make_pair(tag, TransTable::RouteRecord()));
		assert(table.second);
		tab_it = table.first;
		tab_it->second.sender_rank_ = recv_it->second;

		sending_buffs.push_back(TransTable::BufferRecord());
		if(1 < info_it->second.size())
		{
			union_route(node, info_it->second, tab_it->second);

			assert(route_idx_it != route_indice.end());
			for(auto route_idx : route_idx_it->second)
			{
				recving_buffs[route_idx].clear();
			}
			
			
			HIP_CHECK(hipMemcpy(tab_it->second.sender_active_colinds_->mutable_cpu_data(), tab_it->second.sender_active_colinds_->gpu_data(), tab_it->second.sender_active_colinds_->size(), hipMemcpyDeviceToHost));
			
			node.block_->update_F_routing_offsets_gpu(tab_it->second.sender_active_colinds_->gpu_data(),
												tab_it->second.sender_active_colinds_->count(),
												tab_it->second.receiver_colinds_->count(),
												tab_it->second.receiver_colinds_->mutable_gpu_data());
		
			HIP_CHECK(hipDeviceSynchronize());
			
			std::get<0>(sending_buffs.back()) = tab_it->second.sender_active_colinds_->mutable_cpu_data();
			std::get<1>(sending_buffs.back()) = static_cast<int>(tab_it->second.sender_active_colinds_->count());
		}
		else
		{
			assert(1 == info_it->second.size());
			assert(tab_it->second.receiver_ranks_.insert(info_it->second.begin()->first).second);
			std::get<0>(sending_buffs.back()) = std::get<0>(info_it->second.begin()->second);
			std::get<1>(sending_buffs.back()) = std::get<1>(info_it->second.begin()->second);
		}
		
		int err = MPI_Isend(std::get<0>(sending_buffs.back()),
						std::get<1>(sending_buffs.back()),
						MPI_UNSIGNED,
						recv_it->second,
						tag,
						node.info_->comm_,
						&request);
		assert(err == MPI_SUCCESS);
		requests.push_back(request);
	}
}

template<typename T, typename T2>
static void recv_init(NodeInfo<T, T2>& node,
					TransTable::Config& conf,
					vector<MPI_Request>& requests,
					vector<TransTable::BufferRecord>& sending_buffs)
{
	map<unsigned short, tuple<unsigned int*, int>> sending_records;
	map<unsigned short, unsigned int> sending_indice;
	vector<vector<unsigned int>> recving_buffs;
	map<int, set<unsigned int>> route_indice;

	unsigned int recving_count = conf.sending_ranks_.size();
	for(const auto& pair : conf.routing_ranks_)
	{
		assert(pair.first != node.info_->rank_);
		const auto& route = std::get<1>(pair.second);
		if(route.find(node.info_->rank_) != route.end())
		{
			recving_count += (std::get<1>(pair.second).size() - 1);
		}
		else
		{
			recving_count += std::get<1>(pair.second).size();
		}
	}
	recving_buffs.resize(recving_count);

	for(unsigned int idx = 0; idx < recving_count; idx++)
	{	
		int elems, rank, tag, err;
		MPI_Status status;
		
		err = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, node.info_->comm_, &status);
		assert(err == MPI_SUCCESS && status.MPI_SOURCE != node.info_->rank_);
		rank = status.MPI_SOURCE;
		tag = status.MPI_TAG;

		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS);

		if(elems > 0)
		{
			recving_buffs[idx].resize(elems);
		}
		
		err = MPI_Recv(recving_buffs[idx].data(),
					elems,
					MPI_UNSIGNED,
					rank,
					tag,
					node.info_->comm_,
					&status);
		assert(err == MPI_SUCCESS);

		if(node.info_->rank_ == tag)
		{
			auto send_it = conf.sending_ranks_.find(rank);
			assert(send_it != conf.sending_ranks_.end());
			if(elems > 0)
			{
				assert(sending_records.find(RANK2ID(rank)) == sending_records.end());
				auto record = sending_records.emplace(RANK2ID(rank), tuple<unsigned int*, int>());
				assert(record.second);
				std::get<0>(record.first->second) = recving_buffs[idx].data();
				std::get<1>(record.first->second) = elems;

				assert(sending_indice.find(RANK2ID(rank)) == sending_indice.end());
				assert(sending_indice.emplace(RANK2ID(rank), idx).second);
			}
			else
			{
				assert(0 == elems);
				conf.sending_ranks_.erase(send_it);
			}
			
			continue;
		}
		
		if(!conf.routing_ranks_.empty())
		{
			route_init(node, conf, rank, tag, idx, recving_buffs, requests, sending_buffs, route_indice);
		}
		else
		{
			LOG_INFO << "[ERROR]: unexpected point-to-point traffic!" << std::endl;
			assert(0);
		}
	}

	assert(conf.sending_ranks_.size() == sending_records.size());
	node.block_->record_F_sending_actives(sending_records);
	sending_records.clear();
	for(auto it = sending_indice.begin(); it != sending_indice.end(); it++)
	{
		recving_buffs[it->second].clear();
	}
	sending_indice.clear();
	
	assert(conf.sending_ranks_.size() == node.block_->f_sending_bids_.size());
	if(!node.block_->f_sending_bids_.empty())
	{
		node.trans_table_->send_requests_.resize(node.block_->f_sending_bids_.size());
		for(unsigned int i = 0; i < node.block_->f_sending_bids_.size(); i++)
		{
			int receiver_rank = ID2RANK(node.block_->f_sending_bids_[i]);
			assert(receiver_rank != node.info_->rank_);
			assert(conf.sending_ranks_.find(receiver_rank) != conf.sending_ranks_.end());
			node.trans_table_->send_buffs_.push_back(TransTable::SendMetaData());
			auto& meta = node.trans_table_->send_buffs_.back();
			meta.receiver_rank_ = receiver_rank;
			std::get<1>(meta.buffer_) = static_cast<int>(node.block_->f_sending_rowptrs_->cpu_data()[i + 1] - node.block_->f_sending_rowptrs_->cpu_data()[i]);
		}
	}

	if(!conf.recving_ranks_.empty())
	{
		unsigned int route_count = 0;
		unsigned int last_index = 0;
		
		for(auto pair : conf.recving_ranks_)
		{
			auto route_it = node.trans_table_->route_table_.find(pair.first);
			if(route_it != node.trans_table_->route_table_.end())
			{
				unsigned int idx = 0;
				for(auto recv_it = route_it->second.receiver_ranks_.begin();
					recv_it != route_it->second.receiver_ranks_.end();
					recv_it++, idx++)
				{
					if(node.info_->rank_ == *recv_it)
					{
						auto table = node.trans_table_->recv_table_.emplace(std::make_pair(pair.first, TransTable::RecvRecord()));
						assert(table.second);
						
						table.first->second.type_ = TransTable::MemoryType::MEMORY_GPU;
						std::get<0>(table.first->second.record_) = route_it->second.receiver_active_colinds_->mutable_gpu_data() + 
																route_it->second.receiver_rowptrs_->cpu_data()[idx];
						assert(pair.first == ID2RANK(node.block_->f_receiving_bids_[last_index]));
						unsigned int count = route_it->second.receiver_rowptrs_->cpu_data()[idx + 1] - route_it->second.receiver_rowptrs_->cpu_data()[idx];
						assert(count == (node.block_->f_receiving_rowptrs_->cpu_data()[last_index + 1] - node.block_->f_receiving_rowptrs_->cpu_data()[last_index]));
						std::get<1>(table.first->second.record_) = count;
						last_index++;
						break;
					}
				}
				
				route_count++;
			}
			else
			{
				assert(!node.block_->f_receiving_bids_.empty() && last_index < node.block_->f_receiving_bids_.size());
				assert(pair.first == ID2RANK(node.block_->f_receiving_bids_[last_index]));
				int count = static_cast<int>(node.block_->f_receiving_rowptrs_->cpu_data()[last_index + 1] - node.block_->f_receiving_rowptrs_->cpu_data()[last_index]);
				auto table = node.trans_table_->recv_table_.emplace(std::make_pair(pair.first, TransTable::RecvRecord()));
				assert(table.second);
				table.first->second.type_ = TransTable::MemoryType::MEMORY_CPU;
				std::get<1>(table.first->second.record_) = count;
				last_index++;
			}
		}

		assert(route_count == node.trans_table_->route_table_.size() && 
				last_index == node.block_->f_receiving_bids_.size());

		node.trans_table_->recv_requests_.resize(conf.recving_ranks_.size());
	}
	else
	{
		assert(node.block_->f_receiving_bids_.empty() && node.trans_table_->route_table_.empty());
	}

	if(TransTable::Mode::MODE_POINT_TO_POINT != node.trans_table_->mode_)
	{
		unordered_map<unsigned int, shared_ptr<TransTable::RouteRecord::MergeRecord>> merge_records;
		
		//merge acording to the composition of level and receiver_rank.
		int idx = 0;
		for(auto route_it = conf.routing_ranks_.begin(); route_it != conf.routing_ranks_.end(); route_it++)
		{
			for(auto receiver_rank : std::get<1>(route_it->second))
			{
				if(node.info_->rank_ != receiver_rank)
				{
					unsigned int key;
					if(TransTable::Mode::MODE_ROUTE_WITH_MERGE == node.trans_table_->mode_)
					{
						key = (static_cast<unsigned int>(std::get<0>(route_it->second)) << 16) + static_cast<unsigned int>(receiver_rank);
					}
					else
					{
						key = (static_cast<unsigned int>(route_it->first) << 16) + static_cast<unsigned int>(receiver_rank);
					}
					
					auto record_it = merge_records.find(key);
					if(record_it == merge_records.end())
					{
						auto record = merge_records.emplace(key, make_shared<TransTable::RouteRecord::MergeRecord>());
						assert(record.second);
						record_it = record.first;
						record_it->second->index_ = idx++;
						node.trans_table_->route_buffs_.push_back(TransTable::RouteMetaData());
						auto& buff = node.trans_table_->route_buffs_.back();
						buff.offset_ = 0;
						buff.receiver_rank_ = receiver_rank;
						buff.receiver_tag_ = route_it->first;
					}

					assert(record_it->second->receipt_set_.insert(route_it->first).second);	
				}	
			}
		}

		for (auto& record : merge_records)
		{
			auto& buff = node.trans_table_->route_buffs_[record.second->index_];
			
			for(auto receipt_rank : record.second->receipt_set_)
			{
				unsigned int key = (buff.receiver_rank_ << 16) + static_cast<unsigned int>(receipt_rank);
				assert(TransTable::RouteRecord::merge_records_.find(key) == TransTable::RouteRecord::merge_records_.end());
				assert(TransTable::RouteRecord::merge_records_.emplace(key, record.second).second);
			
				auto info_it = conf.routing_infos_.find(receipt_rank);
				assert(info_it != conf.routing_infos_.end());
				auto receiver_it = info_it->second.find(buff.receiver_rank_);
				assert(receiver_it != info_it->second.end());
				
				if(1 < record.second->receipt_set_.size())
				{
					record.second->action_ = TransTable::Action::ACTION_MERGE;
					buff.offset_ += std::get<1>(receiver_it->second) + 2;
				}
				else
				{
					record.second->action_ = TransTable::Action::ACTION_ROUTE;
					record.second->receipt_set_.clear();
					buff.offset_ += std::get<1>(receiver_it->second);
					break;
				}
			}
		}

		if(!node.trans_table_->route_buffs_.empty())
		{
			for(auto& route : node.trans_table_->route_buffs_)
			{
				std::get<0>(route.buffer_).reset(new DataAllocator<unsigned int>(node.info_->rank_, route.offset_ * sizeof(unsigned int), false));
				route.offset_ = 0;
				assert(route.receipt_set_.empty());
			}
			
			node.trans_table_->route_requests_.resize(node.trans_table_->route_buffs_.size());
		}
	}

	if(!requests.empty())
	{
		int err = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
		assert(err == MPI_SUCCESS);	
	}

}

static void parse_merge(const unsigned int* content,
						int content_size,
						unordered_map<int, TransTable::BufferRecord>& records)
{
	int content_read;
	while(content_size > 0) 
	{
		content_read = content[0];
		// modify position to unread content
		content++;
		content_size--;

		assert(content_read > 1 && content_size >= content_read);
		auto out = records.emplace(std::make_pair(content[0], TransTable::BufferRecord()));
		assert(out.second);
		std::get<0>(out.first->second) = const_cast<unsigned int*>(content) + 1;
		std::get<1>(out.first->second) = content_read - 1;
		content +=  content_read;
		content_size -= content_read;
	}
}

template<typename T, typename T2>
static void validate_record(NodeInfo<T, T2>& node,
							const int rank,
							const TransTable::BufferRecord& buff,
							TransTable::RecvMetaData* meta = nullptr,
							TransTable::RecvRecord::MergeRecord* merge_record = nullptr)
{
	
	auto recv_it = node.trans_table_->recv_table_.find(rank);
	assert(recv_it != node.trans_table_->recv_table_.end());

	if(nullptr != meta)
	{
		meta->action_ = TransTable::Action::ACTION_RECORD;
		std::get<0>(recv_it->second.record_) = std::get<0>(meta->buffer_)->mutable_cpu_data();
	}

	if(nullptr != merge_record)
	{
		merge_record->receipt_actions_.push_back(TransTable::Action::ACTION_RECORD);
	}

	thrust::host_vector<unsigned short> h_recving_bids = node.block_->f_receiving_bids_;
	auto it = thrust::find(h_recving_bids.begin(), h_recving_bids.end(), RANK2ID(rank));
	assert(it != h_recving_bids.end());
	unsigned int idx = it - h_recving_bids.begin();
	int count = static_cast<int>(node.block_->f_receiving_rowptrs_->cpu_data()[idx + 1] - node.block_->f_receiving_rowptrs_->cpu_data()[idx]);
	assert(count == std::get<1>(buff));

	thrust::device_vector<unsigned int> d_data1(std::get<1>(buff));
	HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_data1.data()), 
						std::get<0>(buff),
						std::get<1>(buff) * sizeof(unsigned int),
						hipMemcpyHostToDevice));
	assert(thrust::unique(d_data1.begin(), d_data1.end()) == d_data1.end());
	
	thrust::device_vector<unsigned int> d_data2(count);
	HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_data2.data()), 
						node.block_->f_receiving_colinds_->cpu_data() + node.block_->f_receiving_rowptrs_->cpu_data()[idx],
						count * sizeof(unsigned int),
						hipMemcpyHostToDevice));
	assert(thrust::unique(d_data2.begin(), d_data2.end()) == d_data2.end());
	assert(thrust::equal(d_data1.begin(),  d_data1.end(), d_data2.begin()));
	
}

template<typename T, typename T2>
static void validate_route(NodeInfo<T, T2>& node,
							const int rank,
							const TransTable::BufferRecord& buff,
							vector<MPI_Request>& null_requests,
							vector<TransTable::BufferMetaData>& route_buffs,
							TransTable::RecvMetaData* recv_meta = nullptr,
							TransTable::RecvRecord::MergeRecord* merge_record = nullptr)
{
	auto route_it = node.trans_table_->route_table_.find(rank);
	if(route_it == node.trans_table_->route_table_.end())
	{
		validate_record(node, rank, buff, recv_meta, merge_record);
		return;
	}

	if(nullptr != recv_meta)
	{
		recv_meta->action_ = TransTable::Action::ACTION_ROUTE;
	}
	
	if(nullptr != merge_record)
	{
		merge_record->receipt_actions_.push_back(TransTable::Action::ACTION_ROUTE);
	}

	bool has_union = false;
	if(1 < route_it->second.receiver_ranks_.size())
	{
		route_it->second.sender_active_colinds_->free_cpu_data();
		
		//validate union
		assert(static_cast<unsigned int>(std::get<1>(buff)) == route_it->second.sender_active_colinds_->count());
		thrust::device_vector<unsigned int> d_data(std::get<1>(buff));
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_data.data()),
							std::get<0>(buff),
							std::get<1>(buff) * sizeof(unsigned int),
							hipMemcpyHostToDevice));
		assert(thrust::equal(d_data.begin(), d_data.end(), route_it->second.sender_active_colinds_->gpu_data()));
		has_union = true;
	}

	unsigned int idx = 0;
	for(auto rank_it = route_it->second.receiver_ranks_.begin(); 
			rank_it != route_it->second.receiver_ranks_.end();
			rank_it++, idx++)
	{
		assert(*rank_it < node.info_->size_);
		thrust::device_vector<unsigned int> d_receiver_colinds;
		if(has_union)
		{
			unsigned int count = route_it->second.receiver_rowptrs_->cpu_data()[idx + 1] - route_it->second.receiver_rowptrs_->cpu_data()[idx];
			d_receiver_colinds.resize(count);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_receiver_colinds.data()), 
							route_it->second.receiver_colinds_->gpu_data() + route_it->second.receiver_rowptrs_->cpu_data()[idx],
							count * sizeof(unsigned int), hipMemcpyDeviceToDevice));
			
			node.block_->update_F_routing_neuron_ids_gpu(route_it->second.sender_active_colinds_->gpu_data(),
														count,
														thrust::raw_pointer_cast(d_receiver_colinds.data()));
		}
		
		if(*rank_it == node.info_->rank_)
		{
			assert(has_union && !d_receiver_colinds.empty());

			thrust::host_vector<unsigned int> h_receiver_colinds(d_receiver_colinds.size());
			
			HIP_CHECK(hipMemcpy(h_receiver_colinds.data(), 
							thrust::raw_pointer_cast(d_receiver_colinds.data()),
							d_receiver_colinds.size() * sizeof(unsigned int),
							hipMemcpyDeviceToHost));
			validate_record(node, rank, make_tuple(h_receiver_colinds.data(), static_cast<int>(h_receiver_colinds.size())));
			continue;
		}
			
		unsigned int key = (static_cast<unsigned int>(*rank_it) << 16) + static_cast<unsigned int>(rank);
		auto record_it = TransTable::RouteRecord::merge_records_.find(key);
		assert(record_it != TransTable::RouteRecord::merge_records_.end());
		assert(record_it->second->index_ < route_buffs.size());
		auto& route_meta = node.trans_table_->route_buffs_[record_it->second->index_];
		auto& route_buff = route_buffs[record_it->second->index_];
		assert(route_meta.receiver_rank_ == *rank_it);

		//merge data
		if(record_it->second->action_ == TransTable::Action::ACTION_MERGE)
		{	
			auto& receipt_set = record_it->second->receipt_set_;
			assert(!receipt_set.empty() && receipt_set.find(rank) != receipt_set.end());
			
			if(has_union)
			{
				assert(!d_receiver_colinds.empty());
				std::get<0>(route_buff)->mutable_cpu_data()[route_meta.offset_++] = d_receiver_colinds.size() + 1;
				std::get<0>(route_buff)->mutable_cpu_data()[route_meta.offset_++] = rank;
				HIP_CHECK(hipMemcpy(std::get<0>(route_buff)->mutable_cpu_data() + route_meta.offset_, 
								thrust::raw_pointer_cast(d_receiver_colinds.data()),
								d_receiver_colinds.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
				route_meta.offset_ += d_receiver_colinds.size();
			}
			else
			{
				assert(d_receiver_colinds.empty());
				std::get<0>(route_buff)->mutable_cpu_data()[route_meta.offset_++] = std::get<1>(buff) + 1;
				std::get<0>(route_buff)->mutable_cpu_data()[route_meta.offset_++] = rank;
				memcpy(std::get<0>(route_buff)->mutable_cpu_data() + route_meta.offset_, std::get<0>(buff), std::get<1>(buff) * sizeof(unsigned int));
				route_meta.offset_ += static_cast<unsigned int>(std::get<1>(buff));
			}

			assert(route_meta.receipt_set_.insert(rank).second);

			if(rank != route_meta.receiver_tag_)
			{
				MPI_Request request;
				int err = MPI_Isend(nullptr, 
									0,
									MPI_UNSIGNED,
									*rank_it,
									rank,
									node.info_->comm_,
									&request);
				assert(err == MPI_SUCCESS);
				null_requests.push_back(request);
			}
			
			if(route_meta.receipt_set_.size() == receipt_set.size())
			{
				std::get<0>(route_buff)->mutable_cpu_data()[0] = TransTable::PackageType::PACKAGE_MERGE;
				assert(static_cast<unsigned int>(route_meta.offset_) == std::get<0>(route_buff)->count());
				std::get<1>(route_buff) = route_meta.offset_;
				route_meta.offset_ = 0;
				route_meta.receipt_set_.clear();
				
				int err = MPI_Isend(std::get<0>(route_buff)->cpu_data(), 
								std::get<1>(route_buff),
								MPI_UNSIGNED,
								route_meta.receiver_rank_,
								route_meta.receiver_tag_,
								node.info_->comm_,
								&(node.trans_table_->route_requests_[record_it->second->index_]));
				assert(err == MPI_SUCCESS);
			}
		}
		else
		{
			assert(record_it->second->action_ == TransTable::Action::ACTION_ROUTE);
			std::get<0>(route_buff)->mutable_cpu_data()[0] = TransTable::PackageType::PACKAGE_SINGLE;
			if(has_union)
			{
				assert(std::get<0>(route_buff)->count() == d_receiver_colinds.size() + 1);
				HIP_CHECK(hipMemcpy(std::get<0>(route_buff)->mutable_cpu_data() + 1, 
								thrust::raw_pointer_cast(d_receiver_colinds.data()),
								d_receiver_colinds.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
				std::get<1>(route_buff) = static_cast<int>(d_receiver_colinds.size()) + 1;
			}
			else
			{
				assert(std::get<0>(route_buff)->count() == static_cast<unsigned int>(std::get<1>(buff)) + 1);
				memcpy(std::get<0>(route_buff)->mutable_cpu_data() + 1, std::get<0>(buff), std::get<1>(buff) * sizeof(unsigned int));
				std::get<1>(route_buff) = std::get<1>(buff) + 1;
			}
			
			int err = MPI_Isend(std::get<0>(route_buff)->cpu_data(), 
							std::get<1>(route_buff),
							MPI_UNSIGNED,
							route_meta.receiver_rank_,
							route_meta.receiver_tag_,
							node.info_->comm_,
							&(node.trans_table_->route_requests_[record_it->second->index_]));
			assert(err == MPI_SUCCESS);
		}
	}
}

template<typename T, typename T2>
static void validate_merge(NodeInfo<T, T2>& node,
							const int rank,
							const TransTable::BufferRecord& buff,
							vector<MPI_Request>& null_requests,
							vector<TransTable::BufferMetaData>& route_buffs,
							TransTable::RecvMetaData& meta)
{
	assert(node.trans_table_->mode_ == TransTable::Mode::MODE_ROUTE_WITH_MERGE);
	meta.action_ = TransTable::Action::ACTION_MERGE;

	set<int> receipt_set;
	unordered_map<int, TransTable::BufferRecord> records;
	parse_merge(std::get<0>(buff), std::get<1>(buff), records);
	unsigned int count = 0;
	for(auto& record : records)
	{
		assert(receipt_set.insert(record.first).second);
		//data format: length + rank + spike neuron id, that is, plus 2.
		count += (std::get<1>(record.second) + 2);
	}

	assert(count == static_cast<unsigned int>(std::get<1>(buff)));
			
	assert(TransTable::RecvRecord::merge_records_.find(rank) == TransTable::RecvRecord::merge_records_.end());
	auto merge_record = TransTable::RecvRecord::merge_records_.emplace(std::make_pair(rank, TransTable::RecvRecord::MergeRecord()));
	assert(merge_record.second);
	merge_record.first->second.receipt_set_ = receipt_set;

	for(auto receipt_rank : receipt_set)
	{
		const auto& record = records[receipt_rank];
		validate_route(node,
					receipt_rank,
					record,
					null_requests,
					route_buffs,
					nullptr,
					&merge_record.first->second);
	}
}

template<typename T, typename T2>
static void validate_init(NodeInfo<T, T2>& node)
{
	vector<TransTable::BufferMetaData> send_buffs;
	if(!node.trans_table_->send_requests_.empty())
	{
		send_buffs.reserve(node.trans_table_->send_requests_.size());
		for(unsigned int i = 0; i < node.trans_table_->send_requests_.size(); i++)
		{	
			auto& meta = node.trans_table_->send_buffs_[i];
			auto count = node.block_->f_sending_rowptrs_->cpu_data()[i + 1] - node.block_->f_sending_rowptrs_->cpu_data()[i];
			assert(count == static_cast<unsigned int>(std::get<1>(meta.buffer_)));

			send_buffs.push_back(TransTable::BufferMetaData());
			auto& send_buff = send_buffs.back();
			std::get<1>(send_buff) = static_cast<int>(count) + 1;
			std::get<0>(send_buff) = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, std::get<1>(send_buff) * sizeof(unsigned int), false);
			std::get<0>(send_buff)->mutable_cpu_data()[0] = TransTable::PackageType::PACKAGE_SINGLE;
			HIP_CHECK(hipMemcpy(std::get<0>(send_buff)->mutable_cpu_data() + 1,
					node.block_->f_sending_colinds_->gpu_data() + node.block_->f_sending_rowptrs_->cpu_data()[i],
					count * sizeof(unsigned int), hipMemcpyDeviceToHost));
			int err = MPI_Isend(std::get<0>(send_buff)->cpu_data(), 
							std::get<1>(send_buff), 
							MPI_UNSIGNED,
							meta.receiver_rank_,
							node.info_->rank_,
							node.info_->comm_,
							&node.trans_table_->send_requests_[i]);
			assert(err == MPI_SUCCESS);
		}
	}

	vector<MPI_Request> null_requests;
	vector<TransTable::BufferMetaData> route_buffs;
	if(!node.trans_table_->route_requests_.empty())
	{
		route_buffs.resize(node.trans_table_->route_buffs_.size());
		for(unsigned int i = 0; i < node.trans_table_->route_buffs_.size(); i++)
		{
			std::get<0>(route_buffs[i]) = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, 
							(std::get<0>(node.trans_table_->route_buffs_[i].buffer_)->count() + 1) * sizeof(unsigned int), false);

			node.trans_table_->route_buffs_[i].offset_ = 1;
		}
	}
	
	if(!node.trans_table_->recv_requests_.empty())
	{
		int elems, rank, tag, err;
		MPI_Status status;
		vector<unsigned int> recv_buff;
		
		assert(0 == node.trans_table_->recv_buffs_.size());
		
		for(unsigned int idx = 0; idx < node.trans_table_->recv_requests_.size(); idx++)
		{	
			err = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, node.info_->comm_, &status);
			assert(err == MPI_SUCCESS);
			rank = status.MPI_SOURCE;
			tag = status.MPI_TAG;
			assert(rank != node.info_->rank_);

			err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
			assert(err == MPI_SUCCESS);

			if(elems > static_cast<int>(recv_buff.size()))
			{
				recv_buff.resize(elems);
			}
			
			err = MPI_Recv(recv_buff.data(), elems, MPI_UNSIGNED, rank, tag, node.info_->comm_, &status);
			assert(err == MPI_SUCCESS);

			if(0 == elems)
			{
				continue;
			}

			assert(1 < elems);
			node.trans_table_->recv_buffs_.push_back(TransTable::RecvMetaData());
			auto& meta = node.trans_table_->recv_buffs_.back();
			meta.sender_rank_ = rank;
			meta.sender_tag_ = tag;
			std::get<0>(meta.buffer_) = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, (elems - 1) * sizeof(unsigned int), false);

			switch(node.trans_table_->mode_)
			{
				case TransTable::Mode::MODE_ROUTE_WITH_MERGE:

					if(recv_buff[0] == TransTable::PackageType::PACKAGE_MERGE)
					{
						validate_merge(node, tag, make_tuple(recv_buff.data() + 1, (elems - 1)), null_requests, route_buffs, meta);
					}
					else
					{
						validate_route(node, tag, make_tuple(recv_buff.data() + 1, (elems - 1)), null_requests, route_buffs, &meta);
					}
					break;
				case TransTable::Mode::MODE_ROUTE_WITHOUT_MERGE:
					assert(recv_buff[0] == TransTable::PackageType::PACKAGE_SINGLE);
					validate_route(node, tag, make_tuple(recv_buff.data() + 1, (elems - 1)), null_requests, route_buffs, &meta);
				break;
				case TransTable::Mode::MODE_POINT_TO_POINT:
					assert(recv_buff[0] == TransTable::PackageType::PACKAGE_SINGLE);
					validate_record(node, tag, make_tuple(recv_buff.data() + 1, (elems - 1)), &meta);
				break;
				default:
					assert(0);
				break;
			}
		}

		if(node.trans_table_->recv_requests_.size() != node.trans_table_->recv_buffs_.size())
		{
			node.trans_table_->recv_requests_.resize(node.trans_table_->recv_buffs_.size());
		}
	}

	if(!node.trans_table_->send_requests_.empty())
	{
		int err = MPI_Waitall(node.trans_table_->send_requests_.size(), node.trans_table_->send_requests_.data(), MPI_STATUSES_IGNORE);
		assert(err == MPI_SUCCESS);
	}

	if(!node.trans_table_->route_requests_.empty())
	{
		int err = MPI_Waitall(node.trans_table_->route_requests_.size(), node.trans_table_->route_requests_.data(), MPI_STATUSES_IGNORE);
		assert(err == MPI_SUCCESS);
	}

	if(!null_requests.empty())
	{
		int err = MPI_Waitall(null_requests.size(), null_requests.data(), MPI_STATUSES_IGNORE);
		assert(err == MPI_SUCCESS);
	}

	send_buffs.clear();
	route_buffs.clear();
	null_requests.clear();
/*
	for(unsigned int idx = 0; idx < node.trans_table_->send_buffs_.size(); idx++)
	{
		std::get<0>(node.trans_table_->send_buffs_[idx].buffer_)->cpu_data();
	}
*/
	for(unsigned int idx = 0; idx < node.trans_table_->recv_buffs_.size(); idx++)
	{
		std::get<0>(node.trans_table_->recv_buffs_[idx].buffer_)->cpu_data();
	}
/*
	for(unsigned int idx = 0; idx < node.trans_table_->route_buffs_.size(); idx++)
	{
		std::get<0>(node.trans_table_->route_buffs_[idx].buffer_)->cpu_data();
	}
*/
}

template<typename T, typename T2>
static void snn_init(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	assert(MPI_MASTER_RANK != info.rank_);
	node.clear();
	
	LOG_INFO << "==========Before snn init=============" << endl;
	report_dev_info(node.gid_);
	report_mem_info();
	
	vector<char> route_json;
	int mode;
	snn_init(info, node.path_, route_json, node.delta_t_, mode);
	node.trans_table_->mode_ = static_cast<TransTable::Mode>(mode);

	TransTable::Config conf;
	switch(node.trans_table_->mode_)
	{
		case TransTable::Mode::MODE_ROUTE_WITHOUT_MERGE:
		case TransTable::Mode::MODE_ROUTE_WITH_MERGE:
			if(!config_trans_table(route_json, info, conf))
			{
				node.trans_table_->mode_ = TransTable::Mode::MODE_POINT_TO_POINT;
				LOG_INFO << "Rank" << info.rank_ << " config route failed. fall back peer to peer" << std::endl;
			}
			route_json.clear();
			break;
		case TransTable::Mode::MODE_POINT_TO_POINT:
			for(int rank = 0; rank < info.size_; rank++)
			{
				if(MPI_MASTER_RANK == rank || rank == info.rank_)
					continue;

				assert(conf.recving_ranks_.emplace(std::make_pair(rank, rank)).second);
				assert(conf.sending_ranks_.insert(rank).second);
			}
		break;
		default:
			assert(0);
		break;
	}

	LOG_INFO << "Using " << ((node.trans_table_->mode_ == TransTable::Mode::MODE_POINT_TO_POINT)
								? "point-to-point" : ((node.trans_table_->mode_ == TransTable::Mode::MODE_ROUTE_WITHOUT_MERGE)
								? "route without merging" : "route with merging")) << " communication." << endl;

	LOG_INFO << conf;
	
	config_conn_table<T, T2>(node.path_ + string("/block_") + to_string(info.rank_ - 1) + string(".npz"),
							node);

	assert(snn_sync(info) == MPI_SUCCESS);
	
	vector<MPI_Request> requests;
	vector<TransTable::BufferRecord> sending_buffs;
	send_init<T, T2>(node, conf, requests, sending_buffs);
	recv_init<T, T2>(node, conf, requests, sending_buffs);
	conf.clear();
	requests.clear();
	sending_buffs.clear();
	
	assert(snn_sync(info) == MPI_SUCCESS);

	validate_init(node);
	
	node.block_->f_receiving_colinds_->free_cpu_data();

	LOG_INFO << "==========After snn init=============" << endl;
	report_mem_info(&node.used_cpu_mem);
	report_dev_info(node.gid_, &node.used_gpu_mem, &node.total_gpu_mem);
}

template<typename T>
static void snn_run_report(const MPIInfo& info,
							const bool has_freq,
							unsigned int* freqs,
							const bool has_vmean,
							T* vmeans,
							const int* stat_recvcounts,
							const int* stat_displs,
							const bool has_sample,
							char* spikes,
							T* vmembs,
							const int* sample_recvcounts,
							const int* sample_displs,
							const bool has_isynapse,
							T* isynapses)
{
	int err;
	
	if(has_freq)
	{
		assert(NULL != freqs);
		err = snn_gatherv(info, NULL, 0,
				MPI_UNSIGNED, freqs, stat_recvcounts, stat_displs, MPI_UNSIGNED);
		assert(err == MPI_SUCCESS);
	}
	
	if(has_vmean)
	{
		assert(NULL != vmeans);
		switch(sizeof(T))
		{
			case 4:
				err = snn_gatherv(info, NULL, 0,
						MPI_FLOAT, vmeans, stat_recvcounts, stat_displs, MPI_FLOAT);
			break;
			case 8:
				err = snn_gatherv(info, NULL, 0,
						MPI_DOUBLE, vmeans, stat_recvcounts, stat_displs, MPI_DOUBLE);
			break;
			default:
				assert(0);
			break;
				
		}
		assert(err == MPI_SUCCESS);
	}

	if(has_sample)
	{
		assert(NULL != spikes && NULL != vmembs);
		err = snn_gatherv(info, NULL, 0,
				MPI_CHAR, spikes, sample_recvcounts, sample_displs, MPI_CHAR);
		assert(err == MPI_SUCCESS);
		
		switch(sizeof(T))
		{
			case 4:
				err = snn_gatherv(info, NULL, 0,
						MPI_FLOAT, vmembs, sample_recvcounts, sample_displs, MPI_FLOAT);
			break;
			case 8:
				err = snn_gatherv(info, NULL, 0,
						MPI_DOUBLE, vmembs, sample_recvcounts, sample_displs, MPI_DOUBLE);
			break;
			default:
				assert(0);
			break;
		}
		assert(err == MPI_SUCCESS);
	}

	if(has_isynapse)
	{
		assert(NULL != isynapses);
		switch(sizeof(T))
		{
			case 4:
				err = snn_gatherv(info, NULL, 0,
						MPI_FLOAT, isynapses, stat_recvcounts, stat_displs, MPI_FLOAT);
			break;
			case 8:
				err = snn_gatherv(info, NULL, 0,
						MPI_DOUBLE, isynapses, stat_recvcounts, stat_displs, MPI_DOUBLE);
			break;
			default:
				assert(0);
			break;
				
		}
		assert(err == MPI_SUCCESS);
	}
}


template<typename T, typename T2>
static void snn_run_report(NodeInfo<T, T2>& node)
{
	time_point<steady_clock> time_start;
	duration<double> diff;
	int err;
	
	for(int i = 0; i < node.iter_; i++)
	{
		shared_ptr<RunReportInfo<T>> report = node.reporting_queue_.pop();
		assert(nullptr != report);
		
		time_start = steady_clock::now();
		
		if(node.has_freq_)
		{
			assert(nullptr != report->freqs_);
			err = snn_gatherv(*node.info_, report->freqs_->cpu_data(), static_cast<int>(report->freqs_->count()),
					MPI_UNSIGNED, NULL, NULL, NULL, MPI_UNSIGNED);
			assert(err == MPI_SUCCESS);
		}
		
		if(node.has_vmean_)
		{
			assert(nullptr != report->vmeans_);
			switch(sizeof(T))
			{
				case 4:
					err = snn_gatherv(*node.info_, report->vmeans_->cpu_data(), static_cast<int>(report->vmeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT);
				break;
				case 8:
					err = snn_gatherv(*node.info_, report->vmeans_->cpu_data(), static_cast<int>(report->vmeans_->count()),
							MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE);
				break;
				default:
					assert(0);
				break;
					
			}
			assert(err == MPI_SUCCESS);
		}
		
		if(node.has_sample_)
		{
			const char* spikes = NULL;
			int n = 0;
			if(nullptr != node.samples_)
			{
				spikes = report->spikes_->cpu_data();
				n = static_cast<int>(report->spikes_->count());
			}
			
			err = snn_gatherv(*node.info_, spikes, n,
					MPI_CHAR, NULL, NULL, NULL, MPI_CHAR);
			assert(err == MPI_SUCCESS);

			const T* vmembs = NULL;
			n = 0;
			if(nullptr != node.samples_)
			{
				vmembs = report->vmembs_->cpu_data();
				n = static_cast<int>(report->vmembs_->count());
			}
			
			switch(sizeof(T))
			{
				case 4:
					err = snn_gatherv(*node.info_, vmembs, n,
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT);
				break;
				case 8:
					err = snn_gatherv(*node.info_, vmembs, n,
							MPI_DOUBLE, NULL, NULL,	NULL, MPI_DOUBLE);
				break;
				default:
					assert(0);
				break;
			}
			assert(err == MPI_SUCCESS);
		}

		if(node.has_imean_)
		{
			assert(nullptr != report->imeans_);
			switch(sizeof(T))
			{
				case 4:
					err = snn_gatherv(*node.info_, report->imeans_->cpu_data(), static_cast<int>(report->imeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT);
				break;
				case 8:
					err = snn_gatherv(*node.info_, report->imeans_->cpu_data(), static_cast<int>(report->imeans_->count()),
							MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE);
				break;
				default:
					assert(0);
				break;
					
			}
			assert(err == MPI_SUCCESS);
		}
		
		diff = steady_clock::now() - time_start;
		node.reporting_duration_[i] = diff.count();

		LOG_INFO << "the " << i << "th snn_run_report done" << std::endl;
	}
}

static void snn_run(const MPIInfo& info,
					int& iter,
					int& iter_offset,
					bool& has_freq,
					bool& has_vmean,
					bool& has_sample,
					int& send_strategy,
					bool& has_isynapse)
{
	vector<int> record(7);
	int err;
		
	if(MPI_MASTER_RANK == info.rank_)
	{
		record[0] = iter;
		record[1] = iter_offset;
		record[2] = has_freq;
		record[3] = has_vmean;
		record[4] = has_sample;
		record[5] = send_strategy;
		record[6] = has_isynapse;
	}

	err = MPI_Bcast(record.data(), record.size(), MPI_INT, MPI_MASTER_RANK, info.comm_);
	assert(err == MPI_SUCCESS);

	if(MPI_MASTER_RANK != info.rank_)
	{
		iter = record[0];
		iter_offset = record[1];
		has_freq = record[2];
		has_vmean = record[3];
		has_sample = record[4];
		send_strategy = record[5];
		has_isynapse = record[6];
	}
}

template<typename T, typename T2>
static void route_routine(NodeInfo<T, T2>& node,
						const int iter,
						const int rank,
						const TransTable::BufferRecord& buff)
{
	auto route_it = node.trans_table_->route_table_.find(rank);
	assert(route_it != node.trans_table_->route_table_.end());
	bool has_union = false;

	//LOG_INFO << "[route_routine]: " << std::get<1>(buff) << ", tag: " << rank << std::endl;
	if(1 < route_it->second.receiver_ranks_.size())
	{
		//LOG_INFO << "[route_routine]: has union route." << std::endl;
		route_it->second.sender_count_ = static_cast<unsigned int>(std::get<1>(buff));
		assert(route_it->second.sender_count_ <= route_it->second.sender_active_colinds_->count());
		if(route_it->second.sender_count_ > 0)
		{
			auto computing_start = steady_clock::now();
			HIP_CHECK(hipMemcpy(route_it->second.sender_active_colinds_->mutable_gpu_data(), 
									std::get<0>(buff),
									route_it->second.sender_count_ * sizeof(unsigned int),
									hipMemcpyHostToDevice));

			node.block_->update_F_routing_spikes_gpu(route_it->second.sender_active_colinds_->gpu_data(),
													route_it->second.sender_count_,
													route_it->second.f_actives_->count(),
													route_it->second.f_actives_->mutable_gpu_data());
			
			node.block_->update_F_routing_actives_gpu(route_it->second.f_actives_->gpu_data(),
													route_it->second.receiver_rowptrs_->gpu_data(),
													route_it->second.receiver_colinds_->gpu_data(),
													route_it->second.receiver_rowptrs_->count() - 1,
													route_it->second.receiver_colinds_->count(),
													route_it->second.f_receiver_actives_->mutable_gpu_data(),
													route_it->second.receiver_block_rowptrs_->mutable_gpu_data(),
													route_it->second.receiver_active_rowptrs_->mutable_gpu_data(),
													route_it->second.receiver_active_colinds_->mutable_gpu_data());

			
			HIP_CHECK(hipMemcpy(route_it->second.receiver_active_rowptrs_->mutable_cpu_data(), 
									route_it->second.receiver_active_rowptrs_->gpu_data(),
									route_it->second.receiver_rowptrs_->size(),
									hipMemcpyDeviceToHost));
			duration<double> diff = steady_clock::now() - computing_start;
			node.computing_duration_[iter] += diff.count();
			node.routing_duration_[iter] += diff.count();
		}

		has_union = true;
	}
	
	unsigned int i = 0;
	for(auto rank_it = route_it->second.receiver_ranks_.begin(); 
						rank_it != route_it->second.receiver_ranks_.end();
						rank_it++, i++)
	{
		if(node.info_->rank_ == *rank_it)
		{
			assert(has_union);
			auto recv_it = node.trans_table_->recv_table_.find(rank);
			assert(recv_it != node.trans_table_->recv_table_.end());
			assert(recv_it->second.type_ == TransTable::MemoryType::MEMORY_GPU);

			if(0 == std::get<1>(buff))
			{
				std::get<0>(recv_it->second.record_) = nullptr;
				std::get<1>(recv_it->second.record_) = 0;
			}
			else
			{
				std::get<0>(recv_it->second.record_) = route_it->second.receiver_active_colinds_->mutable_gpu_data() + route_it->second.receiver_active_rowptrs_->cpu_data()[i];
				std::get<1>(recv_it->second.record_) = static_cast<int>(route_it->second.receiver_active_rowptrs_->cpu_data()[i + 1] - route_it->second.receiver_active_rowptrs_->cpu_data()[i]);
			}
			continue;
		}
		
		unsigned int key = (static_cast<unsigned int>(*rank_it) << 16) + static_cast<unsigned int>(rank);
		auto record_it = TransTable::RouteRecord::merge_records_.find(key);
		assert(record_it != TransTable::RouteRecord::merge_records_.end());
		auto& route = node.trans_table_->route_buffs_[record_it->second->index_];
		
		switch(record_it->second->action_)
		{
			case TransTable::Action::ACTION_MERGE:
				//LOG_INFO << "[route_routine]: ACTION_MERGE" << std::endl;
				assert(record_it->second->receipt_set_.find(rank) != record_it->second->receipt_set_.end());
				assert(route.receipt_set_.insert(rank).second);
				if(has_union)
				{
					if(0 < std::get<1>(buff))
					{
						int count = static_cast<int>(route_it->second.receiver_active_rowptrs_->cpu_data()[i + 1] - 
									route_it->second.receiver_active_rowptrs_->cpu_data()[i]);
						if(count > 0)
						{
							route.buff_records_.push_back(tuple<int, unsigned int*, int, TransTable::MemoryType>());
							std::get<0>(route.buff_records_.back()) = rank;
							std::get<1>(route.buff_records_.back()) = route_it->second.receiver_active_colinds_->mutable_gpu_data() + route_it->second.receiver_active_rowptrs_->cpu_data()[i];
							std::get<2>(route.buff_records_.back()) = count;
							std::get<3>(route.buff_records_.back()) = TransTable::MemoryType::MEMORY_GPU;
							route.offset_ += (count + 2);
						}
					}
				}
				else
				{
					if(0 < std::get<1>(buff))
					{
						route.buff_records_.push_back(tuple<int, unsigned int*, int, TransTable::MemoryType>());
						std::get<0>(route.buff_records_.back()) = rank;
						std::get<1>(route.buff_records_.back()) = std::get<0>(buff);
						std::get<2>(route.buff_records_.back()) = std::get<1>(buff);
						std::get<3>(route.buff_records_.back()) = TransTable::MemoryType::MEMORY_CPU;
						
						route.offset_ += (std::get<1>(buff) + 2);
					}
				}

				if(record_it->second->receipt_set_.size() == route.receipt_set_.size())
				{
					std::get<1>(route.buffer_) = route.offset_;
					assert(route.offset_ <= static_cast<int>(std::get<0>(route.buffer_)->count()));
					if(route.offset_ > 0)
					{
						//if(route.offset_ != static_cast<int>(std::get<0>(route.buffer_)->count()))
						//{
						//	std::get<0>(route.buffer_).reset(new DataAllocator<unsigned int>(node.info_->rank_, route.offset_ * sizeof(unsigned int), false));
						//}
						int offset = 0;
						assert(!route.buff_records_.empty());
						for(auto& buff_record : route.buff_records_)
						{
							assert(std::get<2>(buff_record) > 0);
							std::get<0>(route.buffer_)->mutable_cpu_data()[offset++] = std::get<2>(buff_record) + 1;
							std::get<0>(route.buffer_)->mutable_cpu_data()[offset++] = std::get<0>(buff_record);
							switch(std::get<3>(buff_record))
							{
								case TransTable::MemoryType::MEMORY_GPU:
									HIP_CHECK(hipMemcpy(std::get<0>(route.buffer_)->mutable_cpu_data() + offset,
															std::get<1>(buff_record),
															std::get<2>(buff_record) * sizeof(unsigned int),
															hipMemcpyDeviceToHost));
								break;
								case TransTable::MemoryType::MEMORY_CPU:
									memcpy(std::get<0>(route.buffer_)->mutable_cpu_data() + offset,
                                            std::get<1>(buff_record),
	                                        std::get<2>(buff_record) * sizeof(unsigned int));
								break;
								default:
									assert(0);
								break;
							}

							offset += std::get<2>(buff_record);
						}
						assert(offset == route.offset_);
					}
					
					route.receipt_set_.clear();
					route.offset_ = 0;
					route.buff_records_.clear();
					node.routing_queue_.push(record_it->second->index_);
				}
			break;
			case TransTable::Action::ACTION_ROUTE:
				//LOG_INFO << "[route_routine]: ACTION_ROUTE" << std::endl;
				assert(record_it->second->receipt_set_.empty());
				if(has_union)
				{
					int count = 0;
					if(0 < std::get<1>(buff))
					{
						count = static_cast<int>(route_it->second.receiver_active_rowptrs_->cpu_data()[i + 1] - 
									route_it->second.receiver_active_rowptrs_->cpu_data()[i]);

						assert(count <= static_cast<int>(std::get<0>(route.buffer_)->count()));
						if(0 < count)
						{
							//std::get<0>(route.buffer_).reset(new DataAllocator<unsigned int>(node.info_->rank_, count * sizeof(unsigned int), false));
							HIP_CHECK(hipMemcpy(std::get<0>(route.buffer_)->mutable_cpu_data(),
													route_it->second.receiver_active_colinds_->gpu_data() + route_it->second.receiver_active_rowptrs_->cpu_data()[i],
													count * sizeof(unsigned int),
													hipMemcpyDeviceToHost));
						}
					}
					
					std::get<1>(route.buffer_) = count;

				}
				else
				{
					if(0 < std::get<1>(buff))
					{
						assert(std::get<1>(buff) <= static_cast<int>(std::get<0>(route.buffer_)->count()));
						//std::get<0>(route.buffer_).reset(new DataAllocator<unsigned int>(node.info_->rank_, std::get<1>(buff) * sizeof(unsigned int), false));
						memcpy(std::get<0>(route.buffer_)->mutable_cpu_data(),
								std::get<0>(buff),
								std::get<1>(buff) * sizeof(unsigned int));
					}
					
					std::get<1>(route.buffer_) = std::get<1>(buff);
					
				}
				
				node.routing_queue_.push(record_it->second->index_);
			break;
			default:
				assert(0);
			break;
		}
	}
	
	//LOG_INFO << "[route_routine]: done ..." << std::endl;
}

template<typename T, typename T2>
static void snn_send(NodeInfo<T, T2>& node,
						std::vector<int>& send_indice, 
						const int iter)
{
	int err;
	
	for(unsigned int j = 0; j < node.trans_table_->send_requests_.size(); j++)
	{
		int index = send_indice[j];
		const auto& meta = node.trans_table_->send_buffs_[index];
		err = MPI_Isend(std::get<0>(meta.buffer_),
						std::get<1>(meta.buffer_),
						MPI_UNSIGNED,
						meta.receiver_rank_,
						node.info_->rank_,
						node.info_->comm_,
						&node.trans_table_->send_requests_[index]);
		assert(err == MPI_SUCCESS);

		if(std::get<1>(meta.buffer_) > 0)
		{
			if(node.rank_in_same_node_.find(meta.receiver_rank_) != node.rank_in_same_node_.end())
			{
				node.sending_byte_size_intra_node_[iter] += static_cast<unsigned int>(std::get<1>(meta.buffer_)) * sizeof(unsigned int);
			}
			else
			{
				node.sending_byte_size_inter_node_[iter] += static_cast<unsigned int>(std::get<1>(meta.buffer_)) * sizeof(unsigned int);
			}
		}
	}

	vector<int> indice(node.trans_table_->send_requests_.size());
	int count;
	duration<double> diff;
	while(true)
    {
		auto time_start = steady_clock::now();
		err = MPI_Waitsome(node.trans_table_->send_requests_.size(),
	                    node.trans_table_->send_requests_.data(),
	                    &count,
	                    indice.data(),
	                    MPI_STATUSES_IGNORE);
	    assert(err == MPI_SUCCESS);

		if(count == MPI_UNDEFINED)
			break;

		diff = steady_clock::now() - time_start;

		for(int j = 0; j < count; j++)
		{
			auto index = indice[j];
			const auto& meta = node.trans_table_->send_buffs_[index];
			if(node.rank_in_same_node_.find(meta.receiver_rank_) != node.rank_in_same_node_.end())
			{
				node.duration_intra_node_[iter] += diff.count();
			}
			else
			{
				node.duration_inter_node_[iter] += diff.count();
			}
		}
	}
}

template<typename T, typename T2>
static void snn_recv(NodeInfo<T, T2>& node, 
						const int iter)
{
	int err;
	for(unsigned int j = 0; j < node.trans_table_->recv_requests_.size(); j++)
	{
		auto& meta = node.trans_table_->recv_buffs_[j];
		err = MPI_Irecv(std::get<0>(meta.buffer_)->mutable_cpu_data(),
						std::get<0>(meta.buffer_)->count(),
						MPI_UNSIGNED,
						meta.sender_rank_,
						meta.sender_tag_,
						node.info_->comm_,
						&node.trans_table_->recv_requests_[j]);
		assert(err == MPI_SUCCESS);
	}

	vector<int> indice(node.trans_table_->recv_requests_.size());
	vector<MPI_Status> status(node.trans_table_->recv_requests_.size());
	int count, elems;
	duration<double> diff;
	while(true)
    {
    	auto time_start = steady_clock::now();
        err = MPI_Waitsome(node.trans_table_->recv_requests_.size(),
                        node.trans_table_->recv_requests_.data(),
                        &count,
                        indice.data(),
                        status.data());
        assert(err == MPI_SUCCESS);

		if(count == MPI_UNDEFINED)
			break;

		diff = steady_clock::now() - time_start;
		for(int j = 0; j < count; j++)
		{
			auto index = indice[j];
			assert(index <  static_cast<int>(node.trans_table_->recv_requests_.size())); 
	        auto& meta = node.trans_table_->recv_buffs_[index];
	        assert(meta.sender_rank_ == status[j].MPI_SOURCE);
	        assert(meta.sender_tag_ == status[j].MPI_TAG);

	        err = MPI_Get_count(&status[j], MPI_UNSIGNED, &elems);
	        assert(err == MPI_SUCCESS && elems <= std::get<0>(meta.buffer_)->count());
	        std::get<1>(meta.buffer_) = elems;

			if(node.rank_in_same_node_.find(meta.sender_rank_) != node.rank_in_same_node_.end())
			{
				if(std::get<1>(meta.buffer_) > 0)
				{
					node.recving_byte_size_intra_node_[iter] += static_cast<unsigned int>(std::get<1>(meta.buffer_)) * sizeof(unsigned int);
				}
				node.duration_intra_node_[iter] += diff.count();
			}
			else
			{
				if(std::get<1>(meta.buffer_) > 0)
				{
					node.recving_byte_size_inter_node_[iter] += static_cast<unsigned int>(std::get<1>(meta.buffer_)) * sizeof(unsigned int);
				}
				
				node.duration_inter_node_[iter] += diff.count();
			}

			node.recving_queue_.push(index);
		}
	}
}

template<typename T, typename T2>
static void snn_route(NodeInfo<T, T2>& node, 
						const int iter)
{
	int err;
	vector<time_point<steady_clock>> routing_start(node.trans_table_->route_requests_.size());
	BlockingQueue<int> queue;
	
	std::thread thrd = std::thread([&node, iter, &queue, &routing_start]()
        {
        	vector<MPI_Request*> requests;
			vector<int> indice;
			size_t count = 0;
			duration<double> diff;
			while(true)
		    {
		    	if(count == requests.size() || !queue.empty())
		    	{
		    		auto index = queue.pop();
					indice.push_back(index);
					requests.push_back(&node.trans_table_->route_requests_[index]);
		    	}

				for(unsigned int j = 0; j < requests.size(); j++)
				{
					auto index = indice[j];
					if (*requests[j] != MPI_REQUEST_NULL)
					{
						int flag = 0;
						assert(MPI_SUCCESS == MPI_Test(requests[j], &flag, MPI_STATUS_IGNORE));

						if(flag)
						{
							count++;
							diff = steady_clock::now() - routing_start[index];

							const auto& meta = node.trans_table_->route_buffs_[index];
							if(node.rank_in_same_node_.find(meta.receiver_rank_) != node.rank_in_same_node_.end())
							{
								node.duration_intra_node_[iter] += diff.count();
							}
							else
							{
								node.duration_inter_node_[iter] += diff.count();
							}
						}
					}
				}

				if(count == node.trans_table_->route_requests_.size())
				{
					break;
				}
			}
        });

	for(unsigned int j = 0; j < node.trans_table_->route_requests_.size(); j++)
	{
		auto index = node.routing_queue_.pop();
		assert(index < static_cast<int>(node.trans_table_->route_requests_.size()));
		auto& meta = node.trans_table_->route_buffs_[index];
		err = MPI_Isend(std::get<0>(meta.buffer_)->cpu_data(),
					std::get<1>(meta.buffer_),
					MPI_UNSIGNED,
					meta.receiver_rank_,
					meta.receiver_tag_,
					node.info_->comm_,
					&node.trans_table_->route_requests_[index]);
		assert(err == MPI_SUCCESS);

		if(std::get<1>(meta.buffer_) > 0)
		{
			if(node.rank_in_same_node_.find(meta.receiver_rank_) != node.rank_in_same_node_.end())
			{
				node.sending_byte_size_intra_node_[iter] += static_cast<unsigned int>(std::get<1>(meta.buffer_)) * sizeof(unsigned int);
			}
			else
			{
				node.sending_byte_size_inter_node_[iter] += static_cast<unsigned int>(std::get<1>(meta.buffer_)) * sizeof(unsigned int);
			}
		}

		routing_start[index] = steady_clock::now();
		queue.push(index);
	}

	if (thrd.joinable()) 
	{
		thrd.join();
	}
}

template<typename T, typename T2>
static void snn_run(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	
	assert(MPI_MASTER_RANK != info.rank_);
	time_point<steady_clock> computing_start;
	duration<double> diff;

	node.block_->get_noise_rate_cpu();
	assert(node.reporting_queue_.empty());
	snn_run(info, node.iter_, node.iter_offset_, node.has_freq_, node.has_vmean_, node.has_sample_, node.send_strategy_, node.has_imean_);
	
	vector<int> send_indice;
	switch(node.send_strategy_)
	{
		case TransTable::SendStrategy::STRATEGY_SEND_SEQUENTIAL:
			send_indice.resize(node.trans_table_->send_requests_.size());
			thrust::sequence(send_indice.data(), send_indice.data() + send_indice.size());
		break;
		
		case TransTable::SendStrategy::STRATEGY_SEND_PAIRWISE:
			node.trans_table_->pairwise(node.info_->rank_, send_indice);
		break;
		
		case TransTable::SendStrategy::STRATEGY_SEND_RANDOM:
			send_indice.resize(node.trans_table_->send_requests_.size());
			thrust::sequence(send_indice.data(), send_indice.data() + send_indice.size());
			node.trans_table_->random_shuffle(send_indice);
		break;
		default:
			assert(0);
		break;
	}
	
	shared_ptr<RunReportInfo<T>> report = make_shared<RunReportInfo<T>>();
	if(node.has_freq_)
	{
		if(nullptr == report->freqs_ || report->freqs_->count() != node.block_->get_total_subblocks())
		{
			report->freqs_.reset(new DataAllocator<unsigned int>(info.rank_, node.block_->get_total_subblocks() * sizeof(unsigned int), false));
		}
		assert(nullptr != report->freqs_);

		report->freqs_->cpu_data();
		node.block_->get_freqs_gpu();
	}
				
	if(node.has_vmean_)
	{
		if(nullptr == report->vmeans_ || report->vmeans_->count() != node.block_->get_total_subblocks())
		{
			report->vmeans_.reset(new DataAllocator<T>(info.rank_, node.block_->get_total_subblocks() * sizeof(T), false));
		}
		assert(nullptr != report->vmeans_);

		report->vmeans_->cpu_data();
		node.block_->get_vmeans_gpu();
	}

	if(node.has_imean_)
	{
		if(nullptr == report->imeans_|| report->imeans_->count() != node.block_->get_total_subblocks())
		{
			report->imeans_.reset(new DataAllocator<T>(info.rank_, node.block_->get_total_subblocks() * sizeof(T), false));
		}
		assert(nullptr != report->imeans_);

		report->imeans_->cpu_data();
		node.block_->get_imeans_gpu();
	}
	
	if(node.has_sample_ && nullptr != node.samples_)
	{
		if(nullptr == report->spikes_ || report->spikes_->size() != node.spikes_->size())
		{
			report->spikes_.reset(new DataAllocator<char>(info.rank_, node.spikes_->size(), false));
		}
		assert(nullptr != report->spikes_);
		report->spikes_->cpu_data();
		node.spikes_->gpu_data();

		if(nullptr == report->vmembs_ || report->vmembs_->size() != node.vmembs_->size())
		{
			report->vmembs_.reset(new DataAllocator<T>(info.rank_, node.vmembs_->size(), false));
		}
		assert(nullptr != report->vmembs_);
		report->vmembs_->cpu_data();
		node.vmembs_->gpu_data();
	}

	node.computing_duration_.resize(node.iter_);
	node.routing_duration_.resize(node.iter_);
	node.reporting_duration_.resize(node.iter_);
	node.duration_inter_node_.resize(node.iter_);
	node.duration_intra_node_.resize(node.iter_);
	
	node.sending_byte_size_inter_node_.resize(node.iter_);
	node.sending_byte_size_intra_node_.resize(node.iter_);
	node.recving_byte_size_inter_node_.resize(node.iter_);
	node.recving_byte_size_intra_node_.resize(node.iter_);

	std::thread sending_thread;
    Notification sending_notification;
    Notification sending_done_notification;
    if(!node.trans_table_->send_requests_.empty())
    {
    	//std::promise<bool> thread_launched;
        sending_thread = std::thread(
                [&node, &send_indice, /*&thread_launched,*/ &sending_notification, &sending_done_notification]()
		{
			//thread_launched.set_value(true);
			for(int i = 0; i < node.iter_; i++)
            {
               sending_notification.Wait();
			   snn_send(node, send_indice, i);
			   sending_done_notification.Notify();
			}
    	});
		//assert(thread_launched.get_future().get());
    }

	std::thread recving_thread;
    Notification recving_notification;

    if(!node.trans_table_->recv_requests_.empty())
    {
    	//std::promise<bool> thread_launched;
        recving_thread = std::thread(
                [&node, /*&thread_launched,*/ &recving_notification]()
        {
        	//thread_launched.set_value(true);
            for(int i = 0; i < node.iter_; i++)
            {
               recving_notification.Wait();
			   snn_recv(node, i);
			}
        });
		//assert(thread_launched.get_future().get());
    }

	std::thread routing_thread;
    Notification routing_notification;
    Notification routing_done_notification;

    if(!node.trans_table_->route_requests_.empty())
    {
    	//std::promise<bool> thread_launched;
        routing_thread = std::thread([&node, /*&thread_launched,*/ &routing_notification, &routing_done_notification]()
        {
        	//thread_launched.set_value(true);
	        for(int i = 0; i < node.iter_; i++)
            {
               routing_notification.Wait();
			   snn_route(node, i);
			   routing_done_notification.Notify();
			}
        });
		//assert(thread_launched.get_future().get());
    }

	assert(snn_sync(info) == MPI_SUCCESS);
	
	node.reporting_notification_.Notify();
	unsigned int timestamp_offset = 0;

	if(node.iter_offset_ > 0)
	{
		if(node.block_->get_input_timestamp_size() > 0)
		{
			thrust::device_vector<unsigned int> d_timestamps(node.block_->get_input_timestamp_size());
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_timestamps.data()), node.block_->get_input_timestamp_data(), node.block_->get_input_timestamp_size() * sizeof(unsigned int), hipMemcpyHostToDevice));
			thrust::device_vector<unsigned int>::iterator it = thrust::lower_bound(d_timestamps.begin(), d_timestamps.end(), static_cast<unsigned int>(node.iter_offset_));
			timestamp_offset = it - d_timestamps.begin();
		}
	}

	for(int i = 0; i < node.iter_; i++)
	{
		node.computing_duration_[i] = 0.;
		
		computing_start = steady_clock::now();
		node.block_->update_time();
		timestamp_offset = node.block_->update_F_input_spike_gpu(static_cast<unsigned int>(i + node.iter_offset_),
																timestamp_offset);	
		node.block_->update_V_membrane_gpu();
	
		if(node.has_freq_)
		{
			node.block_->stat_Freqs_gpu();
		}
		node.block_->update_F_active_gpu(0.f, 1.f, false);
		node.block_->update_F_sending_actives_gpu();
		if(node.has_vmean_)
		{
			node.block_->stat_Vmeans_gpu();
		}
		
		if(node.has_sample_ && nullptr != node.samples_)
		{
			node.block_->stat_Spikes_and_Vmembs_gpu(node.samples_->gpu_data(),
													node.samples_->count(),
													node.spikes_->mutable_gpu_data(),
													node.vmembs_->mutable_gpu_data());

		}
		node.block_->update_J_presynaptic_inner_gpu();
		HIP_CHECK(hipDeviceSynchronize());
		
		diff = steady_clock::now() - computing_start;
		node.computing_duration_[i] = diff.count();

		if(!node.trans_table_->send_requests_.empty())
		{
			HIP_CHECK(hipMemcpy(node.block_->f_sending_active_rowptrs_->mutable_cpu_data(), node.block_->f_sending_active_rowptrs_->gpu_data(), node.block_->f_sending_rowptrs_->size(), hipMemcpyDeviceToHost));
			unsigned int total = node.block_->f_sending_active_rowptrs_->cpu_data()[node.block_->f_sending_rowptrs_->count() - 1];
			if(total > 0)
			{
				HIP_CHECK(hipMemcpy(node.block_->f_sending_active_colinds_->mutable_cpu_data(),
										node.block_->f_sending_active_colinds_->gpu_data(),
										total * sizeof(unsigned int),
										hipMemcpyDeviceToHost));
				
			}
			
			assert((node.block_->f_sending_rowptrs_->count() - 1) == node.trans_table_->send_requests_.size());
			for(unsigned int j = 0; j < node.trans_table_->send_requests_.size(); j++)
			{
				auto& meta = node.trans_table_->send_buffs_[j];
				std::get<0>(meta.buffer_) = node.block_->f_sending_active_colinds_->mutable_cpu_data() + node.block_->f_sending_active_rowptrs_->cpu_data()[j];
				std::get<1>(meta.buffer_) = static_cast<int>(node.block_->f_sending_active_rowptrs_->cpu_data()[j + 1] - node.block_->f_sending_active_rowptrs_->cpu_data()[j]);
			}
		}

		assert(snn_sync(info) == MPI_SUCCESS);

		node.routing_duration_[i] = 0.;
		node.duration_inter_node_[i] = 0.;
		node.duration_intra_node_[i] = 0.;
		node.sending_byte_size_inter_node_[i] = 0;
		node.sending_byte_size_intra_node_[i] = 0;
		node.recving_byte_size_inter_node_[i] = 0;
		node.recving_byte_size_intra_node_[i] = 0;
		if(!node.trans_table_->recv_requests_.empty())
		{
			recving_notification.Notify();
		}

		if(!node.trans_table_->route_requests_.empty())
		{
			routing_notification.Notify();
		}
		
		if(!node.trans_table_->send_requests_.empty())
		{

			sending_notification.Notify();
		}
		
		if(!node.trans_table_->recv_requests_.empty())
		{
			for(unsigned int j = 0; j < node.trans_table_->recv_buffs_.size(); j++)
			{
				auto index = node.recving_queue_.pop();
				const auto& meta = node.trans_table_->recv_buffs_[index];
				assert(index < static_cast<int>(node.trans_table_->recv_buffs_.size()));
				
				switch(meta.action_)
				{
					case TransTable::Action::ACTION_RECORD:
						{
							auto recv_it = node.trans_table_->recv_table_.find(meta.sender_tag_);
							assert(recv_it != node.trans_table_->recv_table_.end());
							assert(std::get<0>(meta.buffer_)->cpu_data() == std::get<0>(recv_it->second.record_));
							std::get<1>(recv_it->second.record_) = std::get<1>(meta.buffer_);
						}
					break;
					case TransTable::Action::ACTION_MERGE:
						{
							assert(node.trans_table_->mode_ == TransTable::Mode::MODE_ROUTE_WITH_MERGE);
							unordered_map<int, TransTable::BufferRecord> merged_records;
							parse_merge(std::get<0>(meta.buffer_)->cpu_data(),
										std::get<1>(meta.buffer_),
										merged_records);
							
							auto merge_it = TransTable::RecvRecord::merge_records_.find(meta.sender_tag_);
							assert(merge_it != TransTable::RecvRecord::merge_records_.end());
							assert(merge_it->second.receipt_actions_.size() == merge_it->second.receipt_set_.size());
							for(auto& record : merged_records)
							{
								assert(merge_it->second.receipt_set_.find(record.first) != merge_it->second.receipt_set_.end());
							}
							
							unsigned int idx = 0;
							for(auto receipt_it = merge_it->second.receipt_set_.begin();
									receipt_it != merge_it->second.receipt_set_.end();
									receipt_it++, idx++)
							{
								auto record_it = merged_records.find(*receipt_it);
								switch(merge_it->second.receipt_actions_[idx])
								{
									case TransTable::Action::ACTION_RECORD:
										{
											auto recv_it = node.trans_table_->recv_table_.find(*receipt_it);
											assert(recv_it != node.trans_table_->recv_table_.end());
											if(record_it != merged_records.end())
											{
												recv_it->second.record_ = record_it->second;
											}
											else
											{
												std::get<0>(recv_it->second.record_) = nullptr;
												std::get<1>(recv_it->second.record_) = 0;
											}
										}
									break;
									case TransTable::Action::ACTION_ROUTE:
										if(record_it != merged_records.end())
										{
											route_routine(node, i, *receipt_it, record_it->second);
										}
										else
										{
											route_routine(node, i, *receipt_it, make_tuple(nullptr, 0));
										}
									break;
									default:
										assert(0);
									break;
								}
							}
						}
					break;
					case TransTable::Action::ACTION_ROUTE:
						route_routine(node, i, meta.sender_tag_, make_tuple(std::get<0>(meta.buffer_)->mutable_cpu_data(), std::get<1>(meta.buffer_)));
					break;
					default:
						assert(0);
					break;
				}
			}
		}	

		if(!node.trans_table_->recv_table_.empty())
		{
			node.block_->f_receiving_active_rowptrs_->mutable_cpu_data()[0] = 0;
			for(unsigned int j = 0; j < node.block_->f_receiving_bids_.size(); j++)
			{
				auto recv_it = node.trans_table_->recv_table_.find(ID2RANK(node.block_->f_receiving_bids_[j]));
				assert(recv_it != node.trans_table_->recv_table_.end());
				unsigned int count = static_cast<unsigned int>(std::get<1>(recv_it->second.record_));
				node.block_->f_receiving_active_rowptrs_->mutable_cpu_data()[j + 1] = node.block_->f_receiving_active_rowptrs_->cpu_data()[j] + count;
				assert(node.block_->f_receiving_active_rowptrs_->cpu_data()[j + 1] <= node.block_->f_receiving_active_colinds_->count());
				
				if(count > 0)
				{
					unsigned int* h_colinds = node.block_->f_receiving_active_colinds_->mutable_cpu_data() + node.block_->f_receiving_active_rowptrs_->cpu_data()[j];
					switch(recv_it->second.type_)
					{
						case TransTable::MemoryType::MEMORY_CPU:
							memcpy(h_colinds, std::get<0>(recv_it->second.record_), count * sizeof(unsigned int));
						break;
						case TransTable::MemoryType::MEMORY_GPU:
							HIP_CHECK(hipMemcpy(h_colinds,
										std::get<0>(recv_it->second.record_),
										count * sizeof(unsigned int),
										hipMemcpyDeviceToHost));
						break;
						default:
							assert(0);
						break;
					}
				}
			}

			unsigned int total = node.block_->f_receiving_active_rowptrs_->cpu_data()[node.block_->f_receiving_active_rowptrs_->count() - 1];
			if(total > 0)
			{
				computing_start = steady_clock::now();
				HIP_CHECK(hipMemcpy(node.block_->f_receiving_active_rowptrs_->mutable_gpu_data(),
										node.block_->f_receiving_active_rowptrs_->cpu_data(),
										node.block_->f_receiving_active_rowptrs_->size(),
										hipMemcpyHostToDevice));
				
				HIP_CHECK(hipMemcpy(node.block_->f_receiving_active_colinds_->mutable_gpu_data(),
											node.block_->f_receiving_active_colinds_->cpu_data(),
											total * sizeof(unsigned int),
											hipMemcpyHostToDevice));

				node.block_->update_F_recving_actives_gpu();
				HIP_CHECK(hipDeviceSynchronize());
				diff = steady_clock::now() - computing_start;
				node.computing_duration_[i] += diff.count();
			}
		}

		if(!node.trans_table_->send_requests_.empty())
		{
			sending_done_notification.Notify();
		}

		if(!node.trans_table_->route_requests_.empty())
		{
			routing_done_notification.Notify();
		}

		//LOG_INFO << "================================================" << std::endl;
		computing_start = steady_clock::now();
		node.block_->update_J_presynaptic_outer_gpu();
		node.block_->update_J_presynaptic_gpu();		
		node.block_->update_I_synaptic_gpu();
		if(node.has_imean_)
		{
			node.block_->stat_Imeans_gpu();
		}
		HIP_CHECK(hipDeviceSynchronize());

		diff = steady_clock::now() - computing_start;
		node.computing_duration_[i] += diff.count();

		if(node.has_freq_)
		{
			HIP_CHECK(hipMemcpy(report->freqs_->mutable_cpu_data(), node.block_->get_freqs_gpu(), report->freqs_->size(), hipMemcpyDeviceToHost));
		}

		if(node.has_vmean_)
		{
			HIP_CHECK(hipMemcpy(report->vmeans_->mutable_cpu_data(), node.block_->get_vmeans_gpu(), report->vmeans_->size(), hipMemcpyDeviceToHost));
		}

		if(node.has_imean_)
		{
			HIP_CHECK(hipMemcpy(report->imeans_->mutable_cpu_data(), node.block_->get_imeans_gpu(), report->imeans_->size(), hipMemcpyDeviceToHost));
		}

		if(node.has_sample_ && nullptr != node.samples_)
		{
			HIP_CHECK(hipMemcpy(report->spikes_->mutable_cpu_data(), node.spikes_->gpu_data(), report->spikes_->size(), hipMemcpyDeviceToHost));
			HIP_CHECK(hipMemcpy(report->vmembs_->mutable_cpu_data(), node.vmembs_->gpu_data(), report->vmembs_->size(), hipMemcpyDeviceToHost));
		}

		node.reporting_queue_.push(report);

		if(i < (node.iter_ - 1))
		{
			assert(snn_sync(info) == MPI_SUCCESS);
		}
	}

	if(!node.trans_table_->send_requests_.empty())
	{
		if (sending_thread.joinable())
		{
			sending_thread.join();
		}
	}

	if(!node.trans_table_->recv_requests_.empty())
	{
		if (recving_thread.joinable())
		{
			recving_thread.join();
		}
	}

	if(!node.trans_table_->route_requests_.empty())
	{
		if (routing_thread.joinable())
		{
			routing_thread.join();
		}
	}

	assert(snn_sync(info) == MPI_SUCCESS);
}

static void snn_name_report(const MPIInfo& info,
							vector<string>& node_names)
{
	int tag = info.size_ + Tag::TAG_REPORT_NAME;
	int err, elems = 0;
	MPI_Status status;
	
	for(int rank = 0; rank < info.size_; rank++)
	{
		if(MPI_MASTER_RANK == rank)
			continue;

		err = MPI_Probe(rank, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_CHAR, &elems);
		assert(err == MPI_SUCCESS);
		
		vector<char> vpath(elems + 1);
		vpath[elems] = 0;
		err = MPI_Recv(vpath.data(), elems, MPI_CHAR, rank, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);

		node_names[rank - 1].assign(vpath.data());
	}
}

static void snn_metric_report(const MPIInfo& info,
							double* computing_duration,
							double* routing_duration,
							double* reporting_duration,
							double* duration_inter_node,
							double* duration_intra_node,
							uint64_t* sending_byte_size_inter_node,
							uint64_t* sending_byte_size_intra_node,
							uint64_t* recving_byte_size_inter_node,
							uint64_t* recving_byte_size_intra_node)
{
	int err;
	MPI_Status status;
	double data;
	uint64_t bytes;

	err = snn_gather(info, &data, 1,
			MPI_DOUBLE, computing_duration, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &data, 1,
			MPI_DOUBLE, routing_duration, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(*node.info_, &node.routing_duration_[i], 1,
				MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
		assert(err == MPI_SUCCESS);

	err = snn_gather(info, &data, 1,
			MPI_DOUBLE, reporting_duration, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &data, 1,
			MPI_DOUBLE, duration_inter_node, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &data, 1,
			MPI_DOUBLE, duration_intra_node, 1, MPI_DOUBLE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &bytes, sizeof(uint64_t),
			MPI_BYTE, sending_byte_size_inter_node, sizeof(uint64_t), MPI_BYTE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &bytes, sizeof(uint64_t),
			MPI_BYTE, sending_byte_size_intra_node, sizeof(uint64_t), MPI_BYTE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &bytes, sizeof(uint64_t),
			MPI_BYTE, recving_byte_size_inter_node, sizeof(uint64_t), MPI_BYTE);
	assert(err == MPI_SUCCESS);

	err = snn_gather(info, &bytes, sizeof(uint64_t),
			MPI_BYTE, recving_byte_size_intra_node, sizeof(uint64_t), MPI_BYTE);
	assert(err == MPI_SUCCESS);
}

template<typename T, typename T2>
static void snn_name_report(NodeInfo<T, T2>& node)
{
	
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_REPORT_NAME;
	int err = MPI_Send(node.name_.c_str(), node.name_.length(), MPI_CHAR, MPI_MASTER_RANK, tag, info.comm_);
	assert(err == MPI_SUCCESS);
}

template<typename T, typename T2>
static void snn_metric_report(NodeInfo<T, T2>& node)
{
	int err;

	for(int i = 0; i < node.iter_; i++)
	{
		err = snn_gather(*node.info_, &node.computing_duration_[i], 1,
				MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
		assert(err == MPI_SUCCESS);

		err = snn_gather(*node.info_, &node.reporting_duration_[i], 1,
				MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
		assert(err == MPI_SUCCESS);

		err = snn_gather(*node.info_, &node.duration_inter_node_[i], 1,
				MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
		assert(err == MPI_SUCCESS);

		err = snn_gather(*node.info_, &node.duration_intra_node_[i], 1,
				MPI_DOUBLE, NULL, 1, MPI_DOUBLE);
		assert(err == MPI_SUCCESS);

		err = snn_gather(*node.info_, &node.sending_byte_size_inter_node_[i], sizeof(uint64_t),
				MPI_BYTE, NULL, sizeof(uint64_t), MPI_BYTE);
		assert(err == MPI_SUCCESS);
		
		err = snn_gather(*node.info_, &node.sending_byte_size_intra_node_[i], sizeof(uint64_t),
				MPI_BYTE, NULL, sizeof(uint64_t), MPI_BYTE);
		assert(err == MPI_SUCCESS);

		err = snn_gather(*node.info_, &node.recving_byte_size_inter_node_[i], sizeof(uint64_t),
				MPI_BYTE, NULL, sizeof(uint64_t), MPI_BYTE);
		assert(err == MPI_SUCCESS);

		err = snn_gather(*node.info_, &node.recving_byte_size_intra_node_[i], sizeof(uint64_t),
				MPI_BYTE, NULL, sizeof(uint64_t), MPI_BYTE);
		assert(err == MPI_SUCCESS);
	}
}

template<typename T>
static void snn_update_prop(const MPIInfo& info,
								const int rank,
								const unsigned int* neuron_indice,
								const unsigned int* prop_indice,
								const T* prop_vals,
								const int n)
{
	int tag = info.size_ + Tag::TAG_UPDATE_PROP;
	int err;
	vector<MPI_Request> requests(3);
	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);
	
	err = MPI_Isend(neuron_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]);
	assert(err == MPI_SUCCESS);
	
	tag++;
	err = MPI_Isend(prop_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[1]);
	assert(err == MPI_SUCCESS);
	
	tag++;
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Isend(prop_vals, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]);
			assert(err == MPI_SUCCESS);
			
		break;
		case 8:
			err = MPI_Isend(prop_vals, n, MPI_DOUBLE, rank, tag, info.comm_, &requests[2]);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}

	#if 0
	if(n > 0)
	{
		vector<T> verified_vals(n);
		MPI_Status status;
		int elems;
		
		tag++;
		switch(sizeof(T))
		{
			case 4:
				err = MPI_Recv(verified_vals.data(), n, MPI_FLOAT, rank, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == n);
			break;
			case 8:
				err = MPI_Recv(verified_vals.data(), n, MPI_DOUBLE, rank, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == n);
			break;
			default:
				assert(0);
			break;
		}

		for(int j = 0 ; j < n; j++)
			assert(verified_vals[j] == prop_vals[j]);
	}
	#endif
	
	err = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
	assert(err == MPI_SUCCESS);
}

template<typename T, typename T2>
static void snn_update_prop(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_PROP;
	int err, elems;
	MPI_Status status;
	
	unique_ptr<DataAllocator<unsigned int>> neuron_indice = nullptr;
	unique_ptr<DataAllocator<unsigned int>> prop_indice = nullptr;
	unique_ptr<DataAllocator<T>> prop_vals = nullptr;

	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
	assert(err == MPI_SUCCESS);
	if(elems > 0)
	{
		neuron_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		err = MPI_Recv(neuron_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == neuron_indice->count());
		HIP_CHECK(hipMemcpy(neuron_indice->mutable_gpu_data(), neuron_indice->cpu_data(), neuron_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		err = MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == 0);
	}

	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
	assert(err == MPI_SUCCESS);
	if(elems > 0)
	{
		prop_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		err = MPI_Recv(prop_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == prop_indice->count());
		HIP_CHECK(hipMemcpy(prop_indice->mutable_gpu_data(), prop_indice->cpu_data(), prop_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		err = MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == 0);
	}

	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Get_count(&status, MPI_FLOAT, &elems);
			assert(err == MPI_SUCCESS);
		break;
		case 8:
			err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}

	if(elems > 0)
	{
		prop_vals = make_unique<DataAllocator<T>>(node.info_->rank_, sizeof(T) * elems);
		switch(sizeof(T))
		{
			case 4:
				err = MPI_Recv(prop_vals->mutable_cpu_data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == prop_vals->count());
			break;
			case 8:
				err = MPI_Recv(prop_vals->mutable_cpu_data(), elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == prop_vals->count());
			break;
			default:
				assert(0);
			break;
		}

		HIP_CHECK(hipMemcpy(prop_vals->mutable_gpu_data(), prop_vals->cpu_data(), prop_vals->size(), hipMemcpyHostToDevice));
	}
	else
	{
		switch(sizeof(T))
		{					
			case 4:
				err = MPI_Recv(NULL, elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			break;
			case 8:
				err = MPI_Recv(NULL, elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			default:
				assert(0);
			break;
		}
	}

	if(elems > 0)
	{
		assert(neuron_indice->count() == elems &&
			prop_indice->count() == elems &&
			prop_vals->count() == elems);
		node.block_->update_Props_gpu(neuron_indice->gpu_data(),
									prop_indice->gpu_data(),
									prop_vals->gpu_data(),
									static_cast<unsigned int>(elems));
		HIP_CHECK(hipDeviceSynchronize());
		
		#if 0
		vector<T> verified_vals;
		node.block_->fetch_props(neuron_indice->cpu_data(),
								prop_indice->cpu_data(),
								static_cast<unsigned int>(elems),
								verified_vals);

		tag++;
		switch(sizeof(T))
		{
			case 4:
				err = MPI_Send(verified_vals.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_);
				assert(err == MPI_SUCCESS);
			break;
			case 8:
				err = MPI_Send(verified_vals.data(), elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_);
				assert(err == MPI_SUCCESS);
			break;
			default:
				assert(0);
			break;
		}
		#endif
	}
}

template<typename T>
static void snn_gamma_report(const MPIInfo& info,
								  const int rank,
								  vector<vector<T>>& prop_vals)
{
	
	int err, elems;
	MPI_Status status;

	for(unsigned int i = 0; i < prop_vals.size(); i++)
	{
		err = MPI_Probe(rank, (int)i, info.comm_, &status);
		assert(err == MPI_SUCCESS);

		switch(sizeof(T))
		{
			case 4:
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS);
			break;
			case 8:
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS);
			break;
			default:
				assert(0);
			break;
		}

		if(elems > 0)
		{
			prop_vals[i].resize(elems);
			
			switch(sizeof(T))
			{
				case 4:
					err = MPI_Recv(prop_vals[i].data(), elems, MPI_FLOAT, rank, (int)i, info.comm_, &status);
					assert(err == MPI_SUCCESS);
					err = MPI_Get_count(&status, MPI_FLOAT, &elems);
					assert(err == MPI_SUCCESS && elems == prop_vals[i].size());
				break;
				case 8:
					err = MPI_Recv(prop_vals.data(), elems, MPI_DOUBLE, rank, (int)i, info.comm_, &status);
					assert(err == MPI_SUCCESS);
					err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
					assert(err == MPI_SUCCESS && elems == prop_vals[i].size());
				break;
				default:
					assert(0);
				break;
			}
		}
		else
		{
			switch(sizeof(T))
			{					
				case 4:
					err = MPI_Recv(NULL, elems, MPI_FLOAT, rank, (int)i, info.comm_, &status);
					assert(err == MPI_SUCCESS);
					err = MPI_Get_count(&status, MPI_FLOAT, &elems);
					assert(err == MPI_SUCCESS && elems == 0);
				break;
				case 8:
					err = MPI_Recv(NULL, elems, MPI_DOUBLE, rank, (int)i, info.comm_, &status);
					assert(err == MPI_SUCCESS);
					err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
					assert(err == MPI_SUCCESS && elems == 0);
				default:
					assert(0);
				break;
			}
		}
	}
}

template<typename T, typename T2>
static void snn_gamma_report(NodeInfo<T, T2>& node)
{
	int err;
	MPIInfo& info = *node.info_;
	
	vector<vector<T>> prop_vals;
	node.block_->fetch_prop_cols(node.prop_indice_.data(),
								node.brain_indice_.data(),
								node.prop_indice_.size(),
								prop_vals);

	assert(node.prop_indice_.size() == prop_vals.size());
	for(unsigned int i = 0; i < prop_vals.size(); i++)
	{
		switch(sizeof(T))
		{
			case 4:
				err = MPI_Send(prop_vals[i].data(), prop_vals[i].size(), MPI_FLOAT, MPI_MASTER_RANK, (int)i, info.comm_);
				assert(err == MPI_SUCCESS);
			break;
			case 8:
				err = MPI_Send(prop_vals[i].data(), prop_vals[i].size(), MPI_DOUBLE, MPI_MASTER_RANK, (int)i, info.comm_);
				assert(err == MPI_SUCCESS);
			break;
			default:
				assert(0);
			break;
		}
	}

	node.prop_indice_.clear();
	node.brain_indice_.clear();
}

template<typename T>
static void snn_update_gamma(const MPIInfo& info,
									const int rank,
									const unsigned int* prop_indice,
									const unsigned int* brain_indice,
									const T* alphas,
									const T* betas,
									const unsigned int n,
									const bool has_prop)
{
	int tag = info.size_ + Tag::TAG_UPDATE_GAMMA;
	int err;
	vector<MPI_Request> requests(5);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);

	err = MPI_Isend(&has_prop, 1, MPI_INT, rank, tag, info.comm_, &requests[0]);
	assert(err == MPI_SUCCESS);
	
	tag++;
	err = MPI_Isend(prop_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[1]);
	assert(err == MPI_SUCCESS);
	
	tag++;
	err = MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[2]);
	assert(err == MPI_SUCCESS);
	
	tag++;
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Isend(alphas, n, MPI_FLOAT, rank, tag, info.comm_, &requests[3]);
			assert(err == MPI_SUCCESS);
		break;
		case 8:
			err = MPI_Isend(alphas, n, MPI_DOUBLE, rank, tag, info.comm_, &requests[3]);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}

	tag++;
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Isend(betas, n, MPI_FLOAT, rank, tag, info.comm_, &requests[4]);
			assert(err == MPI_SUCCESS);
		break;
		case 8:
			err = MPI_Isend(betas, n, MPI_DOUBLE, rank, tag, info.comm_, &requests[4]);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}

	err = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
	assert(err == MPI_SUCCESS);
}

template<typename T, typename T2>
static void snn_update_gamma(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_GAMMA;
	int elems, err, n;
	MPI_Status status;
	
	unique_ptr<DataAllocator<unsigned int>> prop_indice = nullptr;
	unique_ptr<DataAllocator<unsigned int>> brain_indice = nullptr;
	unique_ptr<DataAllocator<T>> alphas = nullptr;
	unique_ptr<DataAllocator<T>> betas = nullptr;

	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_INT, &elems);
	assert(err == MPI_SUCCESS && 1 == elems);
	err = MPI_Recv(&node.has_prop_, elems, MPI_INT, MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_INT, &elems);
	assert(err == MPI_SUCCESS && 1 == elems);
	LOG_INFO << "fetch prop: " << (node.has_prop_ ? "TURE" : "FALSE")<< std::endl;
	
	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
	assert(err == MPI_SUCCESS);
	n = elems;
	
	if(elems > 0)
	{
		node.prop_indice_.resize(elems);
		prop_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		err = MPI_Recv(node.prop_indice_.data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == prop_indice->count());
		HIP_CHECK(hipMemcpy(prop_indice->mutable_gpu_data(), node.prop_indice_.data(), prop_indice->size(), hipMemcpyHostToDevice));

		if(!node.has_prop_)
		{
			node.prop_indice_.clear();
		}
	}
	else
	{
		err = MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == 0);
	}

	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
	assert(err == MPI_SUCCESS);
	assert(n == elems);
	
	if(elems > 0)
	{
		node.brain_indice_.resize(elems);
		brain_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		err = MPI_Recv(node.brain_indice_.data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == brain_indice->count());
		HIP_CHECK(hipMemcpy(brain_indice->mutable_gpu_data(), node.brain_indice_.data(), brain_indice->size(), hipMemcpyHostToDevice));
		if(!node.has_prop_)
		{
			node.brain_indice_.clear();
		}
	}
	else
	{
		err = MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == 0);
	}

	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Get_count(&status, MPI_FLOAT, &elems);
			assert(err == MPI_SUCCESS);
		break;
		case 8:
			err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}
	assert(n == elems);

	if(elems > 0)
	{
		alphas = make_unique<DataAllocator<T>>(node.info_->rank_, sizeof(T) * elems);
		vector<T> buff(elems);
		switch(sizeof(T))
		{					
			case 4:
				err = MPI_Recv(buff.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == alphas->count());
			break;
			case 8:
				err = MPI_Recv(buff.data(), elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == alphas->count());
			break;
			default:
				assert(0);
			break;
		}

		HIP_CHECK(hipMemcpy(alphas->mutable_gpu_data(), buff.data(), alphas->size(), hipMemcpyHostToDevice));
	}
	else
	{
		switch(sizeof(T))
		{					
			case 4:
				err = MPI_Recv(NULL, elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			break;
			case 8:
				err = MPI_Recv(NULL, elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			default:
				assert(0);
			break;
		}
	}

	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Get_count(&status, MPI_FLOAT, &elems);
			assert(err == MPI_SUCCESS);
		break;
		case 8:
			err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}
	assert(n == elems);
	
	if(elems > 0)
	{
		betas = make_unique<DataAllocator<T>>(node.info_->rank_, sizeof(T) * elems);
		vector<T> buff(elems);
		switch(sizeof(T))
		{					
			case 4:
				err = MPI_Recv(buff.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == alphas->count());
			break;
			case 8:
				err = MPI_Recv(buff.data(), elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == alphas->count());
			break;
			default:
				assert(0);
			break;
		}

		HIP_CHECK(hipMemcpy(betas->mutable_gpu_data(), buff.data(), betas->size(), hipMemcpyHostToDevice));
	}
	else
	{
		switch(sizeof(T))
		{					
			case 4:
				err = MPI_Recv(NULL, elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			break;
			case 8:
				err = MPI_Recv(NULL, elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			default:
				assert(0);
			break;
		}
	}

	if(n > 0)
	{
		node.block_->update_Gamma_Prop_Cols_gpu(prop_indice->gpu_data(),
												brain_indice->gpu_data(),
												alphas->gpu_data(),
												betas->gpu_data(),
												n);
		
		HIP_CHECK(hipDeviceSynchronize());
	}
	
}


template<typename T>
static void snn_update_hyperpara(const MPIInfo& info,
									const int rank,
									const unsigned int* prop_indice,
									const unsigned int* brain_indice,
									const T* hyperpara_vals,
									const int n)
{
	int tag = info.size_ + Tag::TAG_UPDATE_HYPERPARA;
	int err;
	vector<MPI_Request> requests(3);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);
	err = MPI_Isend(prop_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]);
	assert(err == MPI_SUCCESS);
	
	tag++;
	err = MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[1]);
	assert(err == MPI_SUCCESS);
	
	tag++;
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Isend(hyperpara_vals, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]);
			assert(err == MPI_SUCCESS);
		break;
		case 8:
			err = MPI_Isend(hyperpara_vals, n, MPI_DOUBLE, rank, tag, info.comm_, &requests[2]);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}
			
	#if 0
	if (n > 0)
	{
		vector<unsigned int> bain_counts;
		vector<T> prop_vals;
		vector<T> verified_vals;
		MPI_Status status;
		int elems;
		
		tag++;
		bain_counts.resize(n);
		err = MPI_Probe(rank, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == n);
		err = MPI_Recv(bain_counts.data(), n, MPI_UNSIGNED, rank, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == n);

		tag++;
		err = MPI_Probe(rank, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);		
		switch(sizeof(T))
		{
			case 4:
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS);
				prop_vals.resize(elems);
				err = MPI_Recv(prop_vals.data(), elems, MPI_FLOAT, rank, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == prop_vals.size());
			break;
			case 8:
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS);
				prop_vals.resize(elems);
				err = MPI_Recv(prop_vals.data(), elems, MPI_DOUBLE, rank, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == prop_vals.size());
			break;
			default:
				assert(0);
			break;
		}

		tag++;
		err = MPI_Probe(rank, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);		
		switch(sizeof(T))
		{
			case 4:
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == prop_vals.size());
				verified_vals.resize(elems);
				err = MPI_Recv(verified_vals.data(), elems, MPI_FLOAT, rank, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == verified_vals.size());
			break;
			case 8:
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == prop_vals.size());
				verified_vals.resize(elems);
				err = MPI_Recv(verified_vals.data(), elems, MPI_DOUBLE, rank, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == verified_vals.size());
			break;
			default:
				assert(0);
			break;
		}

		assert(verified_vals.size() == prop_vals.size());
		
		int k = 0;
		
		assert(k == prop_vals.size());
	}
	#endif
	
	err = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
	assert(err == MPI_SUCCESS);
}

template<typename T, typename T2>
static void snn_update_hyperpara(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_HYPERPARA;
	int elems, err;
	MPI_Status status;
	
	
	unique_ptr<DataAllocator<unsigned int>> prop_indice = nullptr;
	unique_ptr<DataAllocator<unsigned int>> brain_indice = nullptr;
	unique_ptr<DataAllocator<T>> hyperpara_vals = nullptr;
	
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
	assert(err == MPI_SUCCESS);
	if(elems > 0)
	{
		prop_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		err = MPI_Recv(prop_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == prop_indice->count());
		HIP_CHECK(hipMemcpy(prop_indice->mutable_gpu_data(), prop_indice->cpu_data(), prop_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		err = MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == 0);
	}

	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
	assert(err == MPI_SUCCESS);
	if(elems > 0)
	{
		brain_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		err = MPI_Recv(brain_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == brain_indice->count());
		HIP_CHECK(hipMemcpy(brain_indice->mutable_gpu_data(), brain_indice->cpu_data(), brain_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		err = MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == 0);
	}

	tag++;
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	switch(sizeof(T))
	{
		case 4:
			err = MPI_Get_count(&status, MPI_FLOAT, &elems);
			assert(err == MPI_SUCCESS);
		break;
		case 8:
			err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
			assert(err == MPI_SUCCESS);
		break;
		default:
			assert(0);
		break;
	}

	if(elems > 0)
	{
		hyperpara_vals = make_unique<DataAllocator<T>>(node.info_->rank_, sizeof(T) * elems);
		switch(sizeof(T))
		{					
			case 4:
				err = MPI_Recv(hyperpara_vals->mutable_cpu_data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == hyperpara_vals->count());
			break;
			case 8:
				err = MPI_Recv(hyperpara_vals->mutable_cpu_data(), elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == hyperpara_vals->count());
			break;
			default:
				assert(0);
			break;
		}

		HIP_CHECK(hipMemcpy(hyperpara_vals->mutable_gpu_data(), hyperpara_vals->cpu_data(), hyperpara_vals->size(), hipMemcpyHostToDevice));
	}
	else
	{
		switch(sizeof(T))
		{					
			case 4:
				err = MPI_Recv(NULL, elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_FLOAT, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			break;
			case 8:
				err = MPI_Recv(NULL, elems, MPI_DOUBLE, MPI_MASTER_RANK, tag, info.comm_, &status);
				assert(err == MPI_SUCCESS);
				err = MPI_Get_count(&status, MPI_DOUBLE, &elems);
				assert(err == MPI_SUCCESS && elems == 0);
			default:
				assert(0);
			break;
		}
	}

	if(elems > 0)
	{
		assert(prop_indice->count() == elems &&
			brain_indice->count() == elems &&
			hyperpara_vals->count() == elems);
		
		#if 0
		vector<vector<T>> prop_vals;
		vector<vector<T>> verified_vals;
	
		node.block_->fetch_prop_cols(prop_indice->cpu_data(),
								brain_indice->cpu_data(),
								static_cast<unsigned int>(elems),
								prop_vals);
		
		const unsigned int* sbcounts = node.block_->get_sub_bcounts();
		const unsigned int* sbids = node.block_->get_sub_bids();
		for(int i = 0; i < elems; i++)
		{
			const int idx = binary_search((brain_indice->cpu_data())[i], sbids, static_cast<int>(node.block_->get_total_subblocks()));
			if(idx < 0)
				assert(prop_vals[i].empty());
			else
				assert(prop_vals[i].size() == sbcounts[idx]);
		}
		#endif

		node.block_->update_Prop_Cols_gpu(prop_indice->gpu_data(),
									brain_indice->gpu_data(),
									hyperpara_vals->gpu_data(),
									static_cast<unsigned int>(elems));
		HIP_CHECK(hipDeviceSynchronize());

		#if 0
		node.block_->fetch_prop_cols(prop_indice->cpu_data(),
									brain_indice->cpu_data(),
									static_cast<unsigned int>(elems),
									verified_vals);
		assert(verified_vals.size() == prop_vals.size());

		for(int i = 0; i < elems; i++)
		{
			assert(prop_vals[i].size() == verified_vals[i].size());
			for(int j = 0 ; j < prop_vals[i].size(); j++)
			{
				assert(fabs(verified_vals[i][j] - (prop_vals[i][j] * hyperpara_vals->cpu_data()[i])) < 1e-6);
			}
		}
		
		#endif
	}
}

static void snn_update_sample(MPIInfo& info,
								const int rank,
								const unsigned int* sample_indice,
								const int n)
{
	int tag = info.size_ + Tag::TAG_UPDATE_SAMPLE;
	int err;

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);		
	err = MPI_Send(sample_indice, n, MPI_UNSIGNED, rank, tag, info.comm_);
	assert(err == MPI_SUCCESS);
}

template<typename T, typename T2>
static void snn_update_sample(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_SAMPLE;
	int err, elems;
	MPI_Status status;
	
	err = MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status);
	assert(err == MPI_SUCCESS);
	err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
	assert(err == MPI_SUCCESS);
	if(elems > 0)
	{
		if(nullptr == node.samples_)
		{
			node.samples_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		}
		else if(elems != static_cast<int>(node.samples_->count()))
		{
			node.samples_.reset(new DataAllocator<unsigned int>(node.info_->rank_, sizeof(unsigned int) * elems));
		}

		if(nullptr == node.spikes_)
		{
			node.spikes_ = make_unique<DataAllocator<char>>(node.info_->rank_, sizeof(char) * elems);
		}
		else if(elems != static_cast<int>(node.spikes_->count()))
		{
			node.spikes_.reset(new DataAllocator<char>(node.info_->rank_, sizeof(char) * elems));
		}

		if(nullptr == node.vmembs_)
		{
			node.vmembs_ = make_unique<DataAllocator<T>>(node.info_->rank_, sizeof(T) * elems);
		}
		else if(elems != static_cast<int>(node.vmembs_->count()))
		{
			node.vmembs_.reset(new DataAllocator<T>(node.info_->rank_, sizeof(T) * elems));
		}

		err = MPI_Recv(node.samples_->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == node.samples_->count());

		HIP_CHECK(hipMemcpy(node.samples_->mutable_gpu_data(), node.samples_->cpu_data(), node.samples_->size(), hipMemcpyDeviceToHost));

	}
	else
	{
		err = MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status);
		assert(err == MPI_SUCCESS);
		err = MPI_Get_count(&status, MPI_UNSIGNED, &elems);
		assert(err == MPI_SUCCESS && elems == 0);
		
		node.samples_.reset(nullptr);
		node.spikes_.reset(nullptr);
		node.vmembs_.reset(nullptr);
	}
}

static void snn_done(MPIInfo& info, int* total = NULL)
{
	int done = 1;
	MPI_Reduce(&done, total, 1, MPI_INT, MPI_SUM, MPI_MASTER_RANK, info.comm_);
}

class SnnImpl final : public Snn::Service
{
 public:
 	SnnImpl(const int rank, const int size, const int tag, const MPI_Comm comm)
	:info_(new MPIInfo(rank, size, tag, comm)),
	total_subblocks_(0),
	total_samples_(0),
	stat_recvcounts_(size),
	stat_displs_(size),
	sample_recvcounts_(size),
	sample_displs_(size)
 	{
 	}

	Status Init(ServerContext* context, const InitRequest* request, ServerWriter<InitResponse>* writer) override
	{
		string path = request->file_path();
		string route_path = request->route_path();
		float delta_t = request->delta_t();
		int comm_mode = request->mode();
		vector<char> route_json;
		
		cout << "Path: " << path << endl;
		cout << "Route file path: " << route_path << endl;
		cout << "Delta time: " << delta_t << endl;
	
		cout << "Comm mode: " << ((comm_mode == InitRequest::COMM_POINT_TO_POINT)
								? "POINT TO POINT" : ((comm_mode == InitRequest::COMM_ROUTE_WITHOUT_MERGE)
								? "ROUTE WITHOUT MERGING" : "ROUTE WITH MERGING")) << endl;
		
		if(path.empty()||
			(comm_mode != InitRequest::COMM_POINT_TO_POINT &&
			route_path.empty()))
		{
			InitResponse response;
			response.set_status(SnnStatus::SNN_INVALID_PARAMETER);
			writer->Write(response);
			return Status::OK;
		}

		if(comm_mode != InitRequest::COMM_POINT_TO_POINT)
		{
			ifstream ifs;
			ifs.open(route_path);
			if(ifs.is_open())
			{
				ifs.seekg(0, std::ios::end);
				size_t len = ifs.tellg();
				ifs.seekg(0, std::ios::beg);
				route_json.resize(len);
				ifs.read(route_json.data(), len);
				ifs.close();
			}
			else
			{
				InitResponse response;
				response.set_status(SnnStatus::SNN_INVALID_PARAMETER);
				writer->Write(response);
				return Status::OK;
			}
		}
		
		Command cmd = SNN_INIT;
		assert(MPI_SUCCESS == wait_handle(cmd, *info_));
		snn_init(*info_, path, route_json, delta_t, comm_mode);
		assert(MPI_SUCCESS == snn_sync(*info_));
		assert(MPI_SUCCESS == snn_sync(*info_));

		vector<int> neurons_per_block(info_->size_);
		vector<int> subids;
		vector<int> subcounts;
		vector<double> used_cpu_mems(info_->size_);
		vector<double> total_gpu_mems(info_->size_);
		vector<double> used_gpu_mems(info_->size_);

		snn_init_report(*info_,
						stat_recvcounts_.data(),
						stat_displs_.data(),
						neurons_per_block.data(),
						subids,
						subcounts,
						used_cpu_mems.data(),
						total_gpu_mems.data(),
						used_gpu_mems.data());
		
		total_subblocks_ = subids.size();
		cout << "Total subblocks: " << total_subblocks_ << endl;

		int idx = 0;
		for(int i = 1; i < info_->size_; i++)
		{
			InitResponse response;
			response.set_status(SnnStatus::SNN_OK);
			response.set_block_id(i - 1);
			response.set_neurons_per_block(neurons_per_block[i]);
			SubblockInfo* info;
			for(int j = 0; j < stat_recvcounts_[i]; j++)
			{
				info = response.add_subblk_info();
				info->set_subblk_id(subids[idx]);
				info->set_subblk_num(subcounts[idx]);
				idx++;
			}

			response.set_used_cpu_mem(used_cpu_mems[i]);
			response.set_total_gpu_mem(total_gpu_mems[i]);
			response.set_used_gpu_mem(used_gpu_mems[i]);
			writer->Write(response);
		}
		
		return Status::OK;
	}

	Status Run(ServerContext* context, const RunRequest* request, ServerWriter<RunResponse>* writer) override
	{
		double run_duration;
		time_point<steady_clock> time_start;
		duration<double> diff;
		time_start = steady_clock::now();

		iter_ = request->iter();
		int iter_offset = request->iter_offset();
		bool has_freq = request->output_freq();
		bool has_vmean = request->output_vmean();
		bool has_sample = request->output_sample();
		int send_strategy = request->strategy();
		bool has_imean = request->output_imean();
		
		cout << "Iteration: " << iter_ << endl;
		cout << "Iteration offset: " << iter_offset << endl;
		cout << "Output frequencies: " << has_freq << endl;
		cout << "Output vmeans: " << has_vmean << endl;
		cout << "Output samples: " << has_sample << endl;
		cout << "Output imeans: " << has_imean << endl;
		cout << "Send strategy: " << ((send_strategy == RunRequest::STRATEGY_SEND_SEQUENTIAL)
								? "SEQUENCE" : ((send_strategy == RunRequest::STRATEGY_SEND_PAIRWISE)
								? "PAIRWISE" : "RANDOM")) << endl;
		
		if(iter_ <= 0 ||
			((has_freq || has_vmean) && 0 == total_subblocks_) || 
			(has_sample && (0 == total_samples_)))
		{
			RunResponse response;
			response.set_status(SnnStatus::SNN_INVALID_PARAMETER);
			writer->Write(response);
			return Status::OK;
		}
		
		vector<unsigned int> freqs;
		vector<float> vmeans;
		vector<char> spikes;
		vector<float> vmembs;
		vector<float> imeans;
		RunResponse response;
		response.set_status(SnnStatus::SNN_OK);

		if(has_freq)
		{
			assert(total_subblocks_ > 0);
			freqs.resize(total_subblocks_);
			response.add_freq(freqs.data(), freqs.size() * sizeof(unsigned int));
		}

		if(has_vmean)
		{
			assert(total_subblocks_ > 0);
			vmeans.resize(total_subblocks_);
			response.add_vmean(vmeans.data(), vmeans.size() * sizeof(float));
			
		}
		
		if(has_sample)
		{
			assert(total_samples_ > 0);

			spikes.resize(total_samples_);
			response.add_sample_spike(spikes.data(), spikes.size());
			vmembs.resize(total_samples_);
			response.add_sample_vmemb(vmembs.data(), vmembs.size() * sizeof(float));
		}

		if(has_imean)
		{
			assert(total_subblocks_ > 0);
			imeans.resize(total_subblocks_);
			response.add_imean(imeans.data(), imeans.size() * sizeof(float));
			
		}

		diff = steady_clock::now() - time_start;
		run_duration = diff.count();

		std::cout << "===initializing time of run stage: " << run_duration << std::endl;

		time_start = steady_clock::now();
		Command cmd = SNN_RUN;
		assert(MPI_SUCCESS == wait_handle(cmd, *info_));
		snn_run(*info_, iter_, iter_offset, has_freq, has_vmean, has_sample, send_strategy, has_imean);
		assert(MPI_SUCCESS == snn_sync(*info_));
		
		for(int i = 0; i < iter_; i++)
		{
			assert(MPI_SUCCESS == snn_sync(*info_));
			snn_run_report<float>(*info_,
								has_freq,
								freqs.data(),
								has_vmean,
								vmeans.data(),
								stat_recvcounts_.data(),
								stat_displs_.data(),
								has_sample,
								spikes.data(),
								vmembs.data(),
								sample_recvcounts_.data(),
								sample_displs_.data(),
								has_imean,
								imeans.data());
			if(has_freq)
			{
				response.set_freq(0, freqs.data(), freqs.size() * sizeof(unsigned int));
			}

			if(has_vmean)
			{
				response.set_vmean(0, vmeans.data(), vmeans.size() * sizeof(float));
			}
			
			if(has_sample)
			{
				response.set_sample_spike(0, spikes.data(), spikes.size());
				response.set_sample_vmemb(0, vmembs.data(), vmembs.size() * sizeof(float));
			}

			if(has_imean)
			{
				response.set_imean(0, imeans.data(), imeans.size() * sizeof(float));
			}

			if(i < (iter_ - 1))
			{
				writer->Write(response);
				assert(MPI_SUCCESS == snn_sync(*info_));
			}
		}

		diff = steady_clock::now() - time_start;
		run_duration = diff.count();
		std::cout << "===runing time of run stage: " << run_duration << std::endl;
		assert(MPI_SUCCESS == snn_sync(*info_));
		
		writer->Write(response);
		return Status::OK;
	}

	Status Measure(ServerContext* context, const MetricRequest* request, ServerWriter<MetricResponse>* writer) override
	{
		Command cmd = SNN_MEASURE;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		vector<string> node_names(info_->size_ - 1);
		snn_name_report(*info_, node_names);
		
		vector<double> computing_duration(info_->size_);
		vector<double> routing_duration(info_->size_);
		vector<double> reporting_duration(info_->size_);
		vector<double> duration_inter_node(info_->size_);
		vector<double> duration_intra_node(info_->size_);
		vector<uint64_t> sending_byte_size_inter_node(info_->size_);
		vector<uint64_t> sending_byte_size_intra_node(info_->size_);
		vector<uint64_t> recving_byte_size_inter_node(info_->size_);
		vector<uint64_t> recving_byte_size_intra_node(info_->size_);

		for(int i = 0; i < iter_; i++)
		{
			snn_metric_report(*info_,
							computing_duration.data(),
							routing_duration.data(),
							reporting_duration.data(),
							duration_inter_node.data(),
							duration_intra_node.data(),
							sending_byte_size_inter_node.data(),
							sending_byte_size_intra_node.data(),
							recving_byte_size_inter_node.data(),
							recving_byte_size_intra_node.data());
		
			MetricResponse response;
			response.set_status(SnnStatus::SNN_OK);
			for(int j = 1; j < info_->size_; j++)
			{
				MetricInfo* metric = response.add_metric();
				metric->set_name(node_names[j - 1]);
				metric->set_computing_duration(computing_duration[j]);
				metric->set_routing_duration(routing_duration[j]);
				metric->set_reporting_duration(reporting_duration[j]);
				metric->set_duration_inter_node(duration_inter_node[j]);
				metric->set_duration_intra_node(duration_intra_node[j]);
				metric->set_sending_byte_size_inter_node(sending_byte_size_inter_node[j]);
				metric->set_sending_byte_size_intra_node(sending_byte_size_intra_node[j]);
				metric->set_recving_byte_size_inter_node(recving_byte_size_inter_node[j]);
				metric->set_recving_byte_size_intra_node(recving_byte_size_intra_node[j]);
			}
			
			writer->Write(response);
		}
		
		return Status::OK;
	}

	Status Updateprop(ServerContext* context, ServerReader<UpdatePropRequest>* reader,
                     UpdatePropResponse* response) override
    {
		UpdatePropRequest prop;
		vector<int> has_bids(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));
		
		Command cmd = SNN_UPDATE_PROP;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		while (reader->Read(&prop))
		{
			assert(prop.neuron_id_size() == prop.prop_id_size() &&
				prop.neuron_id_size() == prop.prop_val_size());
			
			int bid = prop.block_id();
			int n = prop.neuron_id_size();
			
			if(1 == has_bids[bid])
				continue;

			snn_update_prop<float>(*info_,
							bid + 1,
							reinterpret_cast<const unsigned int*>(prop.neuron_id().data()),
							reinterpret_cast<const unsigned int*>(prop.prop_id().data()),
							prop.prop_val().data(),
							n);
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_prop<float>(*info_,
									i + 1,
									NULL,
									NULL,
									NULL,
									0);
			}
		}
		
		{
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}

		response->set_success(true);
		return Status::OK;
	}

	Status Updategamma(ServerContext* context, ServerReader<UpdateGammaRequest>* reader,
                     UpdateGammaResponse* response) override
    {
		UpdateGammaRequest request;
		vector<int> has_bids(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));

		Command cmd = SNN_UPDATE_GAMMA;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		while (reader->Read(&request))
		{
			int bid = request.block_id();
			assert(bid < (info_->size_ - 1));
			int n = request.prop_id_size();
			assert(n == request.brain_id_size() &&
				n == request.gamma_concentration_size() &&
				n == request.gamma_rate_size());

			if(1 == has_bids[bid])
				continue;
			
			snn_update_gamma<float>(*info_,
								bid + 1,
								reinterpret_cast<const unsigned int*>(request.prop_id().data()),
								reinterpret_cast<const unsigned int*>(request.brain_id().data()),
								request.gamma_concentration().data(),
								request.gamma_rate().data(),
								n,
								false);
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_gamma<float>(*info_,
								i + 1,
								NULL,
								NULL,
								NULL,
								NULL,
								0,
								false);
			}
		}
		
	    {
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		response->set_success(true);
		return Status::OK;
	}

	Status Updategammawithresult(ServerContext* context, ServerReaderWriter<UpdateGammaWithResultResponse, UpdateGammaRequest>* stream) override
	{
		UpdateGammaRequest request;
		vector<int> has_bids(info_->size_ - 1);
		vector<int> counts(info_->size_ - 1);
		vector<vector<unsigned int>> prop_indice(info_->size_ - 1);
		vector<vector<unsigned int>> brain_indice(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));

		Command cmd = SNN_UPDATE_GAMMA;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		while (stream->Read(&request))
		{
			int bid = request.block_id();
			assert(bid < (info_->size_ - 1));
			int n = request.prop_id_size();
			assert(n == request.brain_id_size() &&
				n == request.gamma_concentration_size() &&
				n == request.gamma_rate_size());

			if(1 == has_bids[bid])
				continue;

			if(n > 0)
			{
				prop_indice[bid].resize(n);
				brain_indice[bid].resize(n);
				memcpy(prop_indice[bid].data(), request.prop_id().data(), n * sizeof(unsigned int));
				memcpy(brain_indice[bid].data(), request.brain_id().data(), n * sizeof(unsigned int));
				
				snn_update_gamma<float>(*info_,
									bid + 1,
									prop_indice[bid].data(),
									brain_indice[bid].data(),
									request.gamma_concentration().data(),
									request.gamma_rate().data(),
									n,
									true);
				
				has_bids[bid] = 1;
			}
			counts[bid] = n;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_gamma<float>(*info_,
								i + 1,
								NULL,
								NULL,
								NULL,
								NULL,
								0,
								false);
				counts[i] = 0;
			}
		}
		
		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				assert(0 == counts[i]);
				continue;
			}

			assert(counts[i] > 0);
			vector<vector<float>> prop_vals(counts[i]);
			snn_gamma_report<float>(*info_,
									i + 1,
									prop_vals);
			for(int j = 0; j < counts[i]; j++)
			{
				UpdateGammaWithResultResponse response;
				response.set_block_id(i);
				response.set_prop_id((snn::PropType)prop_indice[i][j]);
				response.set_brain_id((int)brain_indice[i][j]);
				if(!prop_vals[j].empty())
				{
					response.add_prop_val(prop_vals[j].data(), prop_vals[j].size() * sizeof(float));
				}
				stream->Write(response);
			}
		}

		{
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		return Status::OK;
	}

	Status Updatehyperpara(ServerContext* context, ServerReader<UpdateHyperParaRequest>* reader,
                     UpdateHyperParaResponse* response) override
	{
		UpdateHyperParaRequest hp;
		vector<int> has_bids(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));

		Command cmd = SNN_UPDATE_HYPERPARA;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);
		
		while (reader->Read(&hp))
		{
			assert(hp.prop_id_size() == hp.brain_id_size() &&
				hp.prop_id_size() == hp.hpara_val_size());

			int bid = hp.block_id();
			int n = hp.prop_id_size();
			
			if(1 == has_bids[bid])
				continue;
			
			snn_update_hyperpara<float>(*info_,
								bid + 1,
								reinterpret_cast<const unsigned int*>(hp.prop_id().data()),
								reinterpret_cast<const unsigned int*>(hp.brain_id().data()),
								hp.hpara_val().data(),
								n);
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_hyperpara<float>(*info_,
								i + 1,
								NULL,
								NULL,
								NULL,
								0);
			}
		}

		{
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		response->set_success(true);
		return Status::OK;
	}

	Status Updatesample(ServerContext* context, ServerReader<UpdateSampleRequest>* reader,
                     						UpdateSampleResponse* response) override
    {
    	UpdateSampleRequest sample;
		int total = 0;
		vector<int> has_bids(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));
		memset(sample_recvcounts_.data(), 0, sample_recvcounts_.size() * sizeof(int));

		Command cmd = SNN_SAMPLE;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);
			
		while(reader->Read(&sample))
		{
			int bid = sample.block_id();
			assert(bid < (info_->size_ - 1));
			int n = sample.sample_idx_size();
			if(1 == has_bids[bid])
				continue;

			snn_update_sample(*info_,
							bid + 1,
							reinterpret_cast<const unsigned int*>(sample.sample_idx().data()),
							n);
			
			has_bids[bid] = 1;
			sample_recvcounts_[bid + 1] = n;
			total += n;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_sample(*info_, i + 1, NULL, 0);
			}
		}

		thrust::exclusive_scan(sample_recvcounts_.begin(), sample_recvcounts_.end(), sample_displs_.begin());
		total_samples_ = total;
		
		{	
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}

		response->set_success(true);
		return Status::OK;
	}
	
	Status Shutdown(ServerContext* context, const ShutdownRequest* request, ShutdownResponse* response) override
	{
		Command cmd = SNN_SHUTDOWN;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);
		
		cout << "Ready for shutdown server...\n";

		int dones = 0;
		snn_done(*info_, &dones);
		assert(dones == info_->size_);
		
		response->set_shutdown(true);
		promise_.set_value();
		
		return Status::OK;
	}

	void Setserver(shared_ptr<Server> server)
	{
		server_ = server;
	}

	shared_ptr<Server> Getserver()
	{
		return server_;
	}

	std::future<void> Getfuture()
	{
		return std::move(promise_.get_future());
	}
	
  private:
  	std::promise<void> promise_;
	
  	unique_ptr<MPIInfo> info_;
	shared_ptr<Server> server_;
	
	size_t total_subblocks_;
	size_t total_samples_;
	
	vector<int> stat_recvcounts_;
	vector<int> stat_displs_;
	vector<int> sample_recvcounts_;
	vector<int> sample_displs_;

	int iter_;
};

static void server_shutdown(void* arg)
{
	SnnImpl* snn = reinterpret_cast<SnnImpl*>(arg);
	
	 auto shutdown_future = snn->Getfuture();
     if (shutdown_future.valid())
	 {
     	shutdown_future.get();
     }

	std::cout << "shutdown grpc server.\n";
	snn->Getserver()->Shutdown();
}

template<typename T, typename T2>
static void node_handle(void* arg)
{
	NodeInfo<T, T2>* node = reinterpret_cast<NodeInfo<T, T2>*>(arg);
	bool quit = false;
	HIP_CHECK(hipSetDevice(node->gid_));
	do{
		int err = wait_handle(node->cmd_, *node->info_);
		assert(err == MPI_SUCCESS);
		switch(node->cmd_)
		{
			case SNN_INIT:
				snn_init<T, T2>(*node);
			break;
			case SNN_RUN:
				snn_run<T, T2>(*node);
			break;
			case SNN_MEASURE:
			break;
			case SNN_UPDATE_PROP:
				snn_update_prop<T, T2>(*node);
			break;
			case SNN_UPDATE_GAMMA:
				snn_update_gamma<T, T2>(*node);
			break;
			case SNN_UPDATE_HYPERPARA:
				snn_update_hyperpara<T, T2>(*node);
			break;
			case SNN_SAMPLE:
				snn_update_sample<T, T2>(*node);
			break;
			case SNN_SHUTDOWN:
				quit = true;
			break;
			default:
				assert(0);
			break;
		}

		if(SNN_RUN != node->cmd_)
		{
			node->reporting_notification_.Notify();
		}
	}while(!quit);
}

template<typename T, typename T2>
static void report_handle(void* arg)
{
	NodeInfo<T, T2>* node = reinterpret_cast<NodeInfo<T, T2>*>(arg);
	bool quit = false;
	do{
		
		node->reporting_notification_.Wait();
		
		switch(node->cmd_)
		{
			case SNN_INIT:
				snn_init_report<T, T2>(*node);
			break;
			case SNN_RUN:
				snn_run_report<T, T2>(*node);
				assert(node->reporting_queue_.empty());
			break;
			case SNN_MEASURE:
				snn_name_report<T, T2>(*node);
				snn_metric_report<T, T2>(*node);
			break;
			case SNN_UPDATE_GAMMA:
				if(node->has_prop_)
					snn_gamma_report<T, T2>(*node);
			case SNN_UPDATE_PROP:
			case SNN_SAMPLE:
			case SNN_UPDATE_HYPERPARA:
			case SNN_SHUTDOWN:
				snn_done(*node->info_);
				if(SNN_SHUTDOWN == node->cmd_)
					quit = true;
			break;
			default:
				assert(0);
			break;
		}
	}while(!quit);
}

static int get_ipaddr_by_hostname(const char *hostname, char *ip_addr, size_t size)
{
	struct addrinfo *result = NULL, hints;
	int ret = -1;
 
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_DGRAM;
	ret = getaddrinfo(hostname, NULL, &hints, &result);
 
	if (ret == 0)
	{
		struct in_addr addr = ((struct sockaddr_in *)result->ai_addr)->sin_addr;
		const char *re_ntop = inet_ntop(AF_INET, &addr, ip_addr, size);
		if (re_ntop == NULL)
			ret = -1;	
	}
 
	freeaddrinfo(result);
	return ret;
}

static int get_ibaddr_by_name(const char *ibname, char **ip_addr)
{
	int ret = -1;
 
	struct sockaddr_in  sin;
    struct ifreq        ifr;

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
	assert(fd != -1);

	strncpy(ifr.ifr_name, ibname, IFNAMSIZ - 1);      //Interface name

    if(ioctl(fd, SIOCGIFADDR, &ifr) == 0) 
	{
        memcpy(&sin, &ifr.ifr_addr, sizeof(ifr.ifr_addr));
        *ip_addr = inet_ntoa(sin.sin_addr);
		ret = 0;
    } 
	
	return ret;
}

static void get_host_name(char* hostname, int maxlen) 
{
	gethostname(hostname, maxlen);
	for (int i = 0; i < maxlen; i++) 
	{
		if (hostname[i] == '.') 
		{
		    hostname[i] = '\0';
		    return;
		}
	}
}

static uint64_t get_host_hash(const char* string) 
{
	// Based on DJB2, result = result * 33 + char
	uint64_t result = 5381;
	for (int c = 0; string[c] != '\0'; c++)
	{
		result = ((result << 5) + result) + string[c];
	}
	
	return result;
}

int main(int argc, char **argv)
{
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	int tag = 0;
	int gpu_id;
	string device;
	int check_memory = get_cmdline_argint(argc, (const char**)argv, "cm");
	int timeout = get_cmdline_argint(argc, (const char**)argv, "to");
	set<int> rank_in_same_node;

	//double time_start;
	
	init_mpi_env(&argc, &argv, rank, gpu_id, size, device);
	{
		char hostname[1024];
		get_host_name(hostname, sizeof(hostname));

		vector<uint64_t> host_hashs(size);
  
		host_hashs[rank] = get_host_hash(hostname);
		MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hashs.data(), sizeof(uint64_t), MPI_BYTE, comm));
		
		for (int p = 0; p < size; p++) 
		{
			 if (p == rank)
			 {
			 	break;
			 }

			 if (host_hashs[p] == host_hashs[rank] && p != MPI_MASTER_RANK)
			 {
			 	assert(rank_in_same_node.insert(p).second);
			 }
		}

		if(rank != MPI_MASTER_RANK && check_memory)
		{
			HIP_CHECK(hipGetLastError());
			bool ret = report_gpu_mem(rank, hostname, 15.0);
			if(!ret)
			{
				MPI_Abort(comm, MPI_ERR_NO_MEM);
				return 1;
			}
		}

		if(check_memory)
		{
			MPICHECK(MPI_Barrier(comm));
		    bool ret = report_mem(rank, hostname, 88.0);
			MPICHECK(MPI_Barrier(comm));
			if(!ret)
			{
				MPI_Abort(comm, MPI_ERR_NO_MEM);
				return 1;
			}
		}

		cout << "The rank (" << rank << ") within the node " << hostname << "." << endl;
	}

	MPICHECK(MPI_Barrier(comm));
	
	if(rank == MPI_MASTER_RANK)
	{
		assert(-1 == gpu_id && device.empty());
		string server_address;
		{
			char* ipaddr;
			int ret = get_ibaddr_by_name("ib0", &ipaddr);
			assert(!ret);
			server_address = string(ipaddr) + string(":50051");
		}
  		SnnImpl service(rank, size, tag, comm);

		ServerBuilder builder;
		builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
		builder.RegisterService(&service);
		builder.SetMaxMessageSize(INT_MAX);
		shared_ptr<Server> server(builder.BuildAndStart());
		cout << "Server listening on " << server_address << endl;
		service.Setserver(server);
		
		std::thread thrd(server_shutdown, &service);

		server->Wait();
		
		thrd.join();
	}
	else
	{
		shared_ptr<NodeInfo<float, float2>> shrd_node = make_shared<NodeInfo<float, float2>>(rank, size, tag, comm, gpu_id, device, timeout, rank_in_same_node);
		//cout << "The rank (" << rank << ") set timeout " << timeout << " seconds." << endl;
		{
			string logfile;
			char link[1024];
    		char exe_path[1024];
			sprintf(link, "/proc/%d/exe", getpid());
    		int n = readlink(link, exe_path, sizeof(exe_path));
    		exe_path[n] = '\0';
			string str(exe_path);
			n = str.rfind("/");
			logfile = str.substr(0, n) + string("/output_") + to_string(rank - 1) + string(".log");
			cout << "rank " << (rank - 1) << " log file path: " << logfile << endl; 
			Logger::instance(logfile.c_str());
		}
		//LOG_SET_VERBOSE(1);
		
		std::thread thrd(report_handle<float, float2>, shrd_node.get());

		node_handle<float, float2>(shrd_node.get());

		thrd.join();
	}
	
	MPI_Finalize();
	DEVICE_RESET
	return 0;
}

