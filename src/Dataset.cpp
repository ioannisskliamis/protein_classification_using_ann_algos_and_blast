#include <algorithm>
#include <iostream>
#include <cassert>
#include <random>
#include <cstdbool>
#include <unordered_set>
#include <utility>
#include <queue>

#include "Dataset.hpp"
#include "euclidean.hpp"

using namespace std;

// Parse big-endian 32-bit unsigned integer
uint32_t Dataset::parse_be_uint32(ifstream& ifs) {
    char buf[4]; // 4-byte buffer

    // Read 4 bytes (32bit int) from file stream
    ifs.read(buf, 4);

    // Convert bytes from big-endian to a 32bit int from most to less significant byte
    uint32_t result = ((uint8_t)(buf[0]) << 24) | ((uint8_t)(buf[1]) << 16) | ((uint8_t)(buf[2]) << 8) | (uint8_t)(buf[3]);

    return result;
}

// Parse little-endian 32-bit unsigned integer
uint32_t Dataset::parse_le_uint32(ifstream& ifs) {
    char buf[4]; // 4-byte buffer

    // Read 4 bytes (32bit int) from file stream
    ifs.read(buf, 4);

    // Convert bytes from little-endian to a 32bit int from most to less significant byte
    uint32_t result = (uint8_t)(buf[0]) | ((uint8_t)(buf[1]) << 8) | ((uint8_t)(buf[2]) << 16) | ((uint8_t)(buf[3]) << 24);

    return result;
}

// Parse little-endian 32-bit float
float Dataset::parse_le_float(ifstream& ifs) {
    char buf[4]; // 4-byte buffer

    // Read 4 bytes (32-bit float) from file stream
    ifs.read(buf, 4);

    float result;

    // Copy the raw bytes from buffer to result
    memcpy(&result, buf, 4);

    return result;
}

// Load big-endian dataset
vector<vector<float>> Dataset::load_mnist_dataset(string& file_path) {
    // Input file is binary
    ifstream ifs(file_path, ios::binary);

    // If file can't be opened
    if(!ifs.is_open()) {
        throw runtime_error("Couldn't open file: " + file_path);
    }

    // Read the file in the mnist format
    uint32_t magic_no = parse_be_uint32(ifs); // Magic number
    uint32_t num_images = parse_be_uint32(ifs); // Total images in the file
    uint32_t num_rows = parse_be_uint32(ifs); // Rows per image
    uint32_t num_cols = parse_be_uint32(ifs); // Columns per image

    size_t dimensions = (size_t)(num_rows * num_cols); // Dimensions of one flattened image
    

    // This is the whole dataset where each vector of floats is one image
    vector<vector<float>> dataset;

    // Loop through all the images
    for(int i = 0; i < num_images; i++) {
        vector<float> image(dimensions); // Vector to hold current image

        // Loop through the pixels of current image
        for(size_t j = 0; j < dimensions; j++) {
            char byte;
            ifs.read(&byte, 1); // Read and store a pixel

            // Convert pixel value to float and add to vector
            image[j] = (unsigned char)byte; // Cast to unsigned to keep true value
            image[j] /= 255.0; // Normalization
        }
        
        // Add loaded image to the dataset
        dataset.push_back(image);
    }

    return dataset;
}

// Load little-endian dataset
vector<vector<float>> Dataset::load_sift_dataset(string& file_path) {
    // Input file is binary
    ifstream ifs(file_path, ios::binary);

    // If file can't be opened
    if(!ifs.is_open()) {
        throw runtime_error("Couldn't open file: " + file_path);
    }

    // This is the whole dataset where each vector of floats is one image
    vector<vector<float>> dataset;

    // Dimension of vector
    uint32_t dimension;

    // Read the whole file
    while(ifs.read((char*)(&dimension), sizeof(uint32_t))) {
        //dimension = parse_le_uint32(ifs);

        // Vector to hold current image
        vector<float> image(dimension);

        // Loop through the features and store them
        for (size_t i = 0; i < dimension; i++) {
            image[i] = parse_le_float(ifs);
        }

        // Normalization
        double l2_norm = 0.0;
        for (float v : image) {
            l2_norm += v * v;
        }
        l2_norm = sqrt(l2_norm);

        if(l2_norm > 0) {
            for (float& v : image) {
                v /= l2_norm;
            }
        }

        // Add loaded image to the dataset
        dataset.push_back(image);
    }

    return dataset;
}

// Load embeddings file
vector<vector<float>> Dataset::load_embeddings(string& file_path) {
    // Input file is binary
    ifstream ifs(file_path, std::ios::binary);
    // If file can't be opened
    if(!ifs.is_open()) {
        throw runtime_error("Couldn't open file: " + file_path);
    }

    size_t dim = 320;

    ifs.seekg(0, ios::end);
    // File size in bytes
    size_t file_size = ifs.tellg();
    ifs.seekg(0, ios::beg);

    size_t total_floats = file_size / sizeof(float);
    size_t N = total_floats / dim;

    // This is the whole dataset where each vector of floats is one embedding
    vector<vector<float>> dataset(N, vector<float>(dim));

    for (size_t i = 0; i < N; i++) {
        ifs.read((char*)(dataset[i].data()), dim * sizeof(float));
    }

    return dataset;
}

Dataset::Dataset(string& stream, const string& type) {
    
    if (type == "mnist") {
        pictures = load_mnist_dataset(stream);
    } 
    else if (type == "sift") {
        pictures = load_sift_dataset(stream);
    }
    else if (type == "protein") {
        pictures = load_embeddings(stream);
    }
    
}

pair<Dataset, vector<int>> Dataset::createSubset(const int& seed) const{
    size_t dataset_size = this->getRows();
    size_t subset_size = static_cast<size_t>(sqrt(dataset_size));

    std::mt19937                gen(seed);
    uniform_int_distribution<>  dist(0, this->getRows() - 1);

    vector<int> selected(dataset_size);

    for (int i = 0; i < dataset_size; i++) {
        selected[i] = i;
    }

    shuffle(selected.begin(), selected.end(), gen);
    selected.resize(subset_size);

    vector<vector<float>> subset;
    subset.reserve(subset_size);

    for(int i = 0; i < selected.size(); i++) {
        subset.push_back(pictures[selected[i]]);
    }

    return { Dataset(subset, type), selected };
}

vector<Neighbor> Dataset::getKNearestNeighbors(double (*dist_fn)(const vector<float>&, const vector<float>&), 
                                            const vector<float>& query_vec, const int& k) const{
    //vector<Neighbor> nearest_neighbors;

    // Holds all distances into a vector and afterwards does a sorting
    // and keeps the first k ones
    /*for(int i = 0; i < pictures.size(); i++){
        nearest_neighbors.push_back({dist_fn(query_vec, pictures[i]), i});
    }
    
    sort(nearest_neighbors.begin(), nearest_neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.distance_to_neighbor < b.distance_to_neighbor;
    });
    
    if(nearest_neighbors.size() > k)
        nearest_neighbors.resize(k);*/
    priority_queue<Neighbor> pq;
    for(int i = 0; i < pictures.size(); i++) {
        double dist = dist_fn(query_vec, pictures[i]);
        if (pq.size() < k)
            pq.push({ dist, i });
        else if (dist < pq.top().distance_to_neighbor) {
            pq.pop();
            pq.push({ dist, i });
        }
    }

    vector<Neighbor> nearest_neighbors;
    nearest_neighbors.reserve(pq.size());

    while (!pq.empty()) {
        nearest_neighbors.push_back(pq.top());
        pq.pop();
    }

    reverse(nearest_neighbors.begin(), nearest_neighbors.end());

    return nearest_neighbors;
}