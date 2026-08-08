#ifndef __IVF_BASE__
#define __IVF_BASE__

#include <cassert>
#include <fstream>
#include <vector>
#include <utility>

#include "Dataset.hpp"
#include "helperStructs.hpp"
#include "parsingFuncs.hpp"
#include "euclidean.hpp"

class IVFBase {
protected:
	struct clusteringResult {           //clustering result 
        struct cluster{
            std::vector<float> centroid;
            std::vector<int>    img_ids;

            cluster(const std::vector<float>& center): centroid(center){}
        };

        std::vector<cluster> clusters;
        double               cluster_time;
        std::vector<double>  silhouettes;

        void find_silhouettes(const Dataset& input, double (*distance_fn)(const std::vector<float>&, const std::vector<float>&));
    };

    struct CentroidDist {
        double distance;
        int centroid_id;
    };

    const int seed;                     //seed          
    const int k_clusters;               //clusters to be made
    const int n_probe;                  //clusters to be checked during search
    const int neighbors;                //neighbors for search
    const double radius;                //radius of search
    const std::string data_type;        //type of data
    const bool range_search;		    //flag for range search

    Dataset              input;
    Dataset              query;
    Dataset              subset;     	 //subset to build ivf
    std::vector<int>     subset_indices; //indices of subset

    //const string output_file;			 //output file

    double               (*distance_fn)(const std::vector<float>&, const std::vector<float>&);

    std::vector<int> centerInitialization(const Dataset&, const int&) const;
    clusteringResult  LloydsAlgorithm(const Dataset&, const int&) const;

    void print_results(Metrics results, std::string&);

public:
    IVFBase(std::string& input_stream, std::string& query_stream, const int& seed, const int& kclusters,
            const int& nprobe, const int& N, const int& R, const bool& range_search, double (*distance_fn)(const std::vector<float>&, const std::vector<float>&), const std::string&);

    void print_silhouettes(std::string& outstream);

    ~IVFBase() {}
};

#endif