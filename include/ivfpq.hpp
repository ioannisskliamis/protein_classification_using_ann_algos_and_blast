#ifndef __IVFPQ_
#define __IVFPQ__

#include <cassert>
#include <fstream>
#include <vector>
#include <tuple>
#include <utility>

#include "Dataset.hpp"
#include "ivfbase.hpp"
#include "readFuncs.hpp"

class IVFPQ : public IVFBase {
private:
    struct Residuals {                                  //Structs that can be helpful for IVFPQ
        std::vector<float> residual;
        int img_id;
        int closest_centroid;
    };

    struct Subvectors {                                  //For IVFPQ
        std::vector<std::vector<float>> vectors;
        int img_id;
    };

    struct PQEntry {                                     //For IVFPQ
        std::vector<int> code;
        int img_id;
    };

    const int M;
    const int cluster_nbits;

    std::tuple<IVFBase::clusteringResult, std::vector<IVFPQ::PQEntry>, std::vector<std::vector<std::vector<float>>>> buildIVFPQ();          //Build PQ for images and codebook

    //Query function
    Metrics IVFPQQuery(const IVFBase::clusteringResult&, const std::vector<IVFPQ::PQEntry>&, const std::vector<std::vector<std::vector<float>>>& , string&) const;

public:
    IVFPQ(std::string& input_stream, std::string& query_stream, const int& seed, const int& kclusters,
          const int& nprobe, const int& N, const int& R, const int& M, const int& cluster_nbits, const bool& range_search,  
          double (*distance_fn)(const std::vector<float>&, const std::vector<float>&), const std::string&);

    void solve(std::string& outstream){
    	auto tuple = buildIVFPQ();
    	clusteringResult result = get<0>(tuple);
    	std::vector<PQEntry> pqentries = get<1>(tuple);
    	std::vector<std::vector<std::vector<float>>> codebook = get<2>(tuple);
        IVFBase::print_results(IVFPQQuery(result, pqentries, codebook, outstream), outstream);
    }

    ~IVFPQ() {}
};

#endif