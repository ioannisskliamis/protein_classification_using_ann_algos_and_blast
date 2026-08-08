#ifndef __HELPERSTRUCTS__
#define __HELPERSTRUCTS__

#include <vector>

using namespace std;

// Struct to hold distance and id of a neighbor
struct Neighbor {
    double distance_to_neighbor;
    int image_id;

    bool operator<(const Neighbor& other) const {
        return distance_to_neighbor < other.distance_to_neighbor;
    }
};

// Struct to hold results of an algorithm
struct Results {
    int id; // Query id
    vector<Neighbor> approximate_neighbors;
    vector<double> true_neighbors_distances;
    vector<int> range_neighbors;
};

// Struct to hold evaluation metrics of an algorithm
struct Metrics {
    string method;
    double average_AF;
    double recall_at_N;
    double qps;
    double average_approximate_time;
    double average_true_time;
};


#endif // __HELPERSTRUCTS__