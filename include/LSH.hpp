#ifndef __LSH__
#define __LSH__

#include <random>
#include <climits>
#include <chrono>
#include <algorithm>
#include <vector>
#include <unordered_map>

#include "helperStructs.hpp"
#include "euclidean.hpp"
#include "Dataset.hpp"
#include "readFuncs.hpp"


using namespace std;

class LSH{
private:
    // Struct for one Hash function h(v)
    struct RandomProjection {
        vector<double> p; // Random projection vector in normal distribution N(0, 1)
        double t; // Shift by t in uniform distribution [0, w)
    };
    // Struct for one element of the ht bucket
    struct BucketEntry {
        int idx; // Index of data point in the dataset
        int ID; // LSH ID for the combination of h
    };

    const int seed;             // seed
    const int k;                // Number of hash functions per g
    const int L;                // Number of hash tables
    const int N;                // Number of neighbors
    const double w;             // Buckets width
    const double R;             // Search range
    const string data_type;     // Dataset type
    const bool range_search;    // Flag to perform range search or not

    default_random_engine rng;

    Dataset data;
    Dataset queries;
    double (*distance_fn)(const vector<float>&, const vector<float>&);  // Distance function

    vector<vector<RandomProjection>> hash_functions;
    vector<unordered_map<int, vector<BucketEntry>>> hash_tables;
    vector<int> r;                                                      // Stores the uniformly random r_i
    
    void initializeHashFunctions();                                     // Creates L group of k hash functions
    void buildHashTables();                                             // Insert vectors in L hash tables
    Metrics search(string&);                                            // Performs ANN
    void print_results(Metrics, string&);

public:
    // Constructor
    LSH(string& input, string& query, const int& seed, const int& k,
                const int& L, const double& w, const int& N, const double& R, const bool& range, 
                double (*distance_fn)(const vector<float>&, const vector<float>&), string& type);

    // Perform search
    void solve(string& outstream) {
        initializeHashFunctions();
        buildHashTables();
        Metrics results = search(outstream);
        print_results(results, outstream);
    }

    ~LSH() {}
};


#endif