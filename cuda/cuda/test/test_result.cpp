#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include "common.hpp"
#include "util/cnpy.h"
#include "test/check.hpp"
#include "util/cmd_arg.hpp"
#include <cassert>

using namespace istbi;

int main(int argc, char **argv)
{
	unsigned int total_neurons;
	string filename;
	
	char* mpath = NULL;
	char* spath = NULL;
	get_cmdline_argstring(argc, (const char**)argv, "mp", &mpath);
	get_cmdline_argstring(argc, (const char**)argv, "sp", &spath);

	int blks = get_cmdline_argint(argc, (const char**)argv, "bs");
	int check_param = get_cmdline_argint(argc, (const char**)argv, "cp");
	int check_transmission = get_cmdline_argint(argc, (const char**)argv, "ct");
	int check_computation = get_cmdline_argint(argc, (const char**)argv, "cc");

	if(check_param)
	{
		assert(mpath && spath);
		assert(blks);
		check_params<float, float2>(spath, mpath, blks);
	}

	if(check_transmission)
	{
		assert(mpath);
		check_exchange_nid(mpath);
		check_exchange_spike(mpath);
	}

	if(check_computation)
	{
		assert(mpath && spath);	
		assert(blks);
		check_result<float, float2>(mpath, blks, spath);
	}
	return 0;
}
