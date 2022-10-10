#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <iostream>
#include "block.hpp"

namespace istbi {

template<typename T, typename T2>
shared_ptr<BrainBlock<T, T2>> init_brain_block(const char* filename,
												const T delta_t,
												const unsigned int bid,
												const unsigned int gid);

template<typename T, typename T2>
void check_params(BrainBlock<T, T2>* block, const char* filename);

template<typename T, typename T2>
void check_params(const char* spath, const char* mpath, const unsigned int blocks);


template<typename T>
void read_samples_from_preset(const char* filename,
					vector<shared_ptr<DataAllocator<T>>>& samples);

template<typename T, typename T2>
void check_result(BrainBlock<T, T2>* block, const char* filename, const unsigned int iter, std::ostream& out = std::cout);

template<typename T, typename T2>
void check_result(BrainBlock<T, T2>* block, const unsigned int iter, std::ostream& out = std::cout);

template<typename T, typename T2>
void check_result(const char* mb_filepath, const unsigned int blks, const char* sb_filepath, std::ostream& out = std::cout);

void check_exchange_spike(const char* path, std::ostream& out = std::cout);

void check_exchange_nid(const char* path, std::ostream& out = std::cout);
} //namespace istbi
