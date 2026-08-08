#include <cassert>
#include <cmath>

#include "euclidean.hpp"

using namespace std;

// Function to caclulate euclidean distance between two vectors
double EuclideanDistance(const vector<float>& v1, const vector<float>& v2){
    assert(v1.size() == v2.size());

    double sum = 0.0;

    size_t dimensionality = v1.size();

    for(int i = 0; i < dimensionality; i++){
		double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}
