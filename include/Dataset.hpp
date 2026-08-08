#ifndef __DATASET__
#define __DATASET__

#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <numeric>
#include <utility>

#include "helperStructs.hpp"

// This structure is used just to hold some vectors.
class Dataset {
private:
    std::vector<std::vector<float>> pictures;
    std::string type;

    // Parse big-endian 32-bit unsigned integer
    uint32_t parse_be_uint32(std::ifstream& ifs);

    // Parse little-endian 32-bit unsigned integer
    uint32_t parse_le_uint32(std::ifstream& ifs);

    // Parse little-endian 32-bit float
    float parse_le_float(std::ifstream& ifs);

    // Load big-endian dataset
    std::vector<std::vector<float>> load_mnist_dataset(std::string& file_path);

    // Load little-endian dataset
    std::vector<std::vector<float>> load_sift_dataset(std::string& file_path);

    // Load embeddings file
    std::vector<std::vector<float>> load_embeddings(std::string& file_path);

public:
    Dataset() = default;                            //Default constructor (we need it when we initialize an object of class IVFBase or it's children)

    Dataset(std::string&, const std::string&);

    Dataset(const std::vector<std::vector<float>>& vectors, const std::string& type) : pictures(vectors), type(type) {}

    int getRows() const{ 
        return pictures.size(); 
    }

    int getColumns() const{ 
        return pictures[0].size(); 
    }

    void resize(int size) {
        this->pictures.resize(size);
    }

    const std::vector<std::vector<float>> getPictures() const{
        return pictures;
    }

    std::pair<Dataset, std::vector<int>> createSubset(const int&) const;

    std::vector<Neighbor> getKNearestNeighbors(double (*dist_fn)(const std::vector<float>&, const std::vector<float>&),
                                            const std::vector<float>&, const int&) const;

    const std::vector<float>& operator[](const int& i) const{
        return pictures[i];
    }

    ~Dataset(){}
};

#endif