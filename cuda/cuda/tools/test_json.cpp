#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <json/json.h>
#include <cassert>
#include <string>

using namespace std;

int main(int argc, char** argv)
{
    ifstream ifs;
    ifs.open(argv[1]);
    assert(ifs.is_open());

	Json::CharReaderBuilder builder;
	builder["collectComments"] = false;
    Json::Value root;
	JSONCPP_STRING errs;

    if (!Json::parseFromStream(builder, ifs, &root, &errs))
    {
        cout << errs << endl;
        return EXIT_FAILURE;
    }

    cout << "total " << root.size() << " elements" << endl;

	auto members = root.getMemberNames();
	for(auto& m : members)
		cout << m << " ";
	cout << endl;

	
	
    for (Json::ArrayIndex i = 0; i < members.size(); ++i)
    {	
    	const Json::Value obj = root[members[i]];
		const Json::Value src = obj["src"];
		const Json::Value dst = obj["dst"];
		assert(src.size() == dst.size());
		char *end;
		cout << "route " << std::strtoul(members[i].c_str(), &end, 10) << endl;
		cout << "src:"<< endl;
		
		for(int j = 0; j < src.size(); j++)
		{
			cout << src[j].asInt() << " ";
		}
		cout << endl;

        cout << "dst:"<< endl;
	    for(int j = 0; j < dst.size(); j++)
		{
			cout << dst[j].asInt() << " ";
		}
		cout << endl;
    }

    return 0;
}

