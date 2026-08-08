#ifndef __HYPERCUBE__
#define __HYPERCUBE__

#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
#include <unordered_map>

#include "helperStructs.hpp"
#include "euclidean.hpp"
#include "Dataset.hpp"
#include "readFuncs.hpp"


using namespace std;

class Hypercube {
private:
    const int seed;               // seed
    const int kproj;              // Number of projection bits (d')
    const int M;                  // Max candidates to check
    const int probes;             // Number of vertices to probe
    const int N;                  // Number of neighbors
    const double w;               // Cell width
    const double R;               // Search range
    const string data_type;       // Dataset type
    const bool range_search;      // Flag to perform range search or not

    default_random_engine rng;

    Dataset data;
    Dataset queries;
    double (*distance_fn)(const vector<float>&, const vector<float>&);  // Distance function

    vector<vector<double>> random_vectors;
    vector<double> t_values;                                            // Offsets
    unordered_map<int, int> h_to_bit;                                   // Bitmapping for hash values
    unordered_map<uint64_t, vector<int>> cube;

    void initializeRandomVectors();                                     // Generate kproj random vectors
    void buildCube();                                                   // Assign each dataset vector to a binary code in the hypercube
    vector<uint64_t> getProbeVertices(uint64_t query_vertex);           // Returns a list of hypercube vertices to probe for a given vertex
    uint64_t computeVertex(vector<float>& vec);                         // Computes the hypercube vertex corresponding to given vector
    Metrics search(string&);                                            // Performs ANN
    void print_results(Metrics, string&);

public:
    // Constructor
    Hypercube(string& input, string& query, const int& seed, const int& kproj,
                const double& w, const int& M, const int& probes, const int& N, const double& R, const bool& range, 
                double (*distance_fn)(const vector<float>&, const vector<float>&), string& type);

    // Perform search
    void solve(string& outstream) {
        initializeRandomVectors();
        buildCube();
        Metrics results = search(outstream);
        print_results(results, outstream);
    }

    ~Hypercube() {}
};

#endif