#ifndef __IVFFLAT__
#define __IVFFLAT__

#include <cassert>
#include <fstream>
#include <vector>
#include <utility>

#include "Dataset.hpp"
#include "ivfbase.hpp"
#include "readFuncs.hpp"

class IVFFlat : public IVFBase {
private:
	IVFBase::clusteringResult buildIVFFlat();					//Builds IVFFlat

	Metrics IVFFlatQuery(const IVFBase::clusteringResult&, string& outstream) const;			//Query function

	void IVFFlatQueryGraph(const IVFBase::clusteringResult& clusteringResult, string&) const; 	// Query function for Project2 to skip brute force

public:
	IVFFlat(std::string& input_stream, std::string& query_stream, const int& seed, const int& kclusters, const int& nprobe, 
		    const int& N, const int& R, const bool& range_search, double (*distance_fn)(const std::vector<float>&, const std::vector<float>&), const std::string&);

	void solve(std::string& outstream){
		auto result = buildIVFFlat();
        IVFBase::print_results(IVFFlatQuery(result, outstream), outstream);
    }

	void create_graph(std::string& outstream){
		auto result = buildIVFFlat();
        IVFFlatQueryGraph(result, outstream);
	}

	~IVFFlat() {}
};

#endif