#include "Hypercube.hpp"

// Constructor
Hypercube::Hypercube(string& input, string& query, const int& seed, const int& kproj,
                const double& w, const int& M, const int& probes, const int& N, const double& R, const bool& range, 
                double (*distance_fun)(const vector<float>&, const vector<float>&), string& type) : data(input, type),
                queries(query, type), seed(seed), kproj(kproj), w(w), M(M), probes(probes), N(N), 
                R(R), range_search(range), distance_fn(distance_fun), data_type(type) {}


// Generate kproj random vectors
void Hypercube::initializeRandomVectors() {
    // Take the dimensions of a vector
    size_t vector_dim = data.getColumns();
    random_vectors.resize(kproj, vector<double>(vector_dim));

    normal_distribution<double> normal(0.0, 1.0);
    uniform_real_distribution<double> uniform(0.0, 1.0);

    // Create kproj random projection vectors
    for (int i = 0; i < kproj; i++) {
        for (size_t dim = 0; dim < vector_dim; dim++) {
            // Each element of the projection vector is a random Gaussian var
            random_vectors[i][dim] = normal(rng);
        }
    }
    
    t_values.resize(kproj);
    // Create a set of random shift "t" values
    for (int i = 0; i < kproj; i++) {
        t_values[i] = uniform(rng) * w;
    }
}


// Assign each dataset vector to a binary code in the hypercube
void Hypercube::buildCube() {
    // Assign each data point to a vertex in the hypercube
    for (int idx = 0; idx < data.getRows(); idx++) {
        uint64_t vertex = 0; // Start with all bits 0

        // Compute kproj individual hash values for this point
        for (int hi = 0; hi < kproj; hi++) {
            // We follow the same formula h(p) = (p * v + t)/w
            double proj = inner_product(data[idx].begin(), data[idx].end(), random_vectors[hi].begin(), 0.0);
            int hval = floor((proj + t_values[hi]) / w);

            // Assign or retrieve bit mapping for current hval
            int bit;
            auto it = h_to_bit.find(hval);
            if (it != h_to_bit.end()) {
                // Use existing mapping if hash already exists
                bit = it->second;
            }
            else {
                // Otherwise assign a new random binary value
                uniform_int_distribution<int> uniform_bit(0, 1);
                int b = uniform_bit(rng);
                h_to_bit[hval] = b;
                bit = b;
            }
            // Insert this bit into the vertex bitstring
            vertex |= ((uint64_t)bit << hi);
        }
        // Add data index to the hypercube vertex bucket
        cube[vertex].push_back(idx);
    }
}


// Returns a list of hypercube vertices to probe for a given vertex
vector<uint64_t> Hypercube::getProbeVertices(uint64_t query_vertex) {
    vector<uint64_t> probe_vertices{query_vertex};

    // Loop over increasing Hamming distances until "probes" parameter
    for (int dist = 1; dist <= probes; dist++) {
        // Prepare bit index list [0, 1, 2, ... kproj-1]
        vector<int> bit_indices(kproj);
        iota(bit_indices.begin(), bit_indices.end(), 0);

        // Boolean selection vector to choose which bits to flip
        vector<bool> select(kproj, false);
        // For Hammind distance = "dist", exactly "dist" bits are set to true (to flip)
        fill(select.begin(), select.begin() + dist, true);

        // Compute the corresponding vertex by flipping those bits in the query's binary code
        do {
            uint64_t flipped = query_vertex;

            // Flip each selected bit using XOR
            for (int i = 0; i < kproj; i++) {
                if (select[i]) {
                    // 1ULL << i creates a bitmask for the i-th bit position
                    // XOR flips that bit in "flipped" var
                    flipped ^= (1ULL << i);
                }
            }
            // Store the new generated neighbor vertex
            probe_vertices.push_back(flipped);

        } while(prev_permutation(select.begin(), select.end())); // Iterate through all dist-bit combinations
    }
    return probe_vertices;
}


// Computes the hypercube vertex corresponding to given vector
uint64_t Hypercube::computeVertex(vector<float>& vec) {
    uint64_t vertex = 0;
    // Loop over each of the kproj functions
    for (int hi = 0; hi < kproj; hi++) {
        double proj = inner_product(vec.begin(), vec.end(), random_vectors[hi].begin(), 0.0);
        int hval = floor((proj + t_values[hi]) / w);

        // Assign or retrieve bit mapping for current hval
        int bit;
        auto it = h_to_bit.find(hval);
        if (it != h_to_bit.end()) {
            // Use existing mapping if hash already exists
            bit = it->second;
        }
        else {
            // Otherwise assign a new random binary value
            uniform_int_distribution<int> uniform_bit(0, 1);
            int b = uniform_bit(rng);
            h_to_bit[hval] = b;
            bit = b;
        }
        // Add this bit to the query's vertex code
        vertex |= ((uint64_t)bit << hi);
    }

    return vertex;
}


// Performs ANN
Metrics Hypercube::search(string& output_file) {
    ofstream ofs(output_file);
    // Metrics
    double total_approx_time = 0.0, total_true_time = 0.0, total_AF = 0.0, total_recall = 0.0;

    auto [protein_id_index, protein_ids] = readProteinIdMap("protein_ids.txt");
    auto [query_id_index, query_ids] = readQueryIdMap("targets.fasta");

    unordered_map<int, vector<int>> blast_hits = readTSV("blast_results.tsv", protein_id_index , query_id_index, N);
    unordered_map<int,unordered_map<int, double>> blast_identities = readIdentities("blast_results.tsv", protein_id_index , query_id_index);

    // For each query in the dataset
    for (int q = 0; q < queries.getRows(); q++) {
        auto query = queries[q];

        uint64_t query_vertex = computeVertex(query);

        auto t0 = chrono::system_clock::now();

        // Get Hamming neighbors
        vector<uint64_t> probe_vertices = getProbeVertices(query_vertex);

        // We now have all binary vertex codes withing Hamming distance <= "probes" from the query vertex
        // so we collect the candidates from cube vertices
        vector<int> candidates;
        vector<int> range_neighbors_ids;
        for (int i = 0; i < probe_vertices.size(); i++) {
            uint64_t vertex = probe_vertices[i];
            
            // If this vertex exists in the hypercube hash table
            if(cube.count(vertex)) {
                // Grab the list of data point indices stored at this vertex
                vector<int>& bucket = cube[vertex];

                // Iterate the data points in this bucket
                for (int j = 0; j < bucket.size(); j++) {
                    candidates.push_back(bucket[j]);
                    // Stop if parameter "M" is reached
                    if(candidates.size() >= M) {
                        break;
                    }
                }
            }
            // Stop probing vertices if parameter "M" is reached
            if(candidates.size() >= M) {
                break;
            }
        }
        if(candidates.empty()) continue;

        // Keep unique candidates
        sort(candidates.begin(), candidates.end());
        candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());

        // Sort candidates by L2 distance (ascending)
        vector<Neighbor> candidate_neighbors;
        for (int id : candidates) {
            double distance = distance_fn(query, data[id]);
            candidate_neighbors.push_back({distance, id});
            // Range search
            if(range_search && distance <= R) {
                range_neighbors_ids.push_back(id);
            }
        }
        sort(candidate_neighbors.begin(), candidate_neighbors.end());

        // Find (at most) N approximate neighbors
        vector<Neighbor> approximate_neighbors;
        for (int i = 0; i < N && i < candidate_neighbors.size(); i++) {
            approximate_neighbors.push_back({candidate_neighbors[i].distance_to_neighbor, candidate_neighbors[i].image_id});
        }

        // Approximate distnace timer
        auto t1 = chrono::system_clock::now();
        total_approx_time += chrono::duration<double>(t1 - t0).count();

        // Perform exhaustive search
        vector<Neighbor> all_true_neighbors;
        vector<int> true_ids;
        vector<double> true_distances;
        
        auto t2 = chrono::system_clock::now();
        all_true_neighbors = data.getKNearestNeighbors(distance_fn, query, N);

        // Retrieve ids and distances of N nearest neighbors
        for (int i = 0; i < all_true_neighbors.size(); i++) {
            true_distances.push_back(all_true_neighbors[i].distance_to_neighbor);
            true_ids.push_back(all_true_neighbors[i].image_id);
        }

        // True distance (exhaustive search) timer
        auto t3 = chrono::system_clock::now();
        total_true_time += chrono::duration<double>(t3 - t2).count();

        // Evaluation of current query
        if(!approximate_neighbors.empty() && !true_distances.empty()) {
            total_AF += approximate_neighbors[0].distance_to_neighbor / true_distances[0]; // Approximation factor
        }

        // Find true neighbors among approximates
        auto& blast_ids = blast_hits[q];

        if (blast_ids.empty()) {
            total_recall += 0.0;
        }

        unordered_set<int> blast_set(blast_ids.begin(), blast_ids.begin() + min(N, (int)blast_ids.size()));      
        int hits_per_query = 0;
        
        for(int k = 0; k < min(N, (int)approximate_neighbors.size()); k++) {
            if (blast_set.count(approximate_neighbors[k].image_id)) {
                hits_per_query++;
            }
        }

        double query_recall = (double) hits_per_query / min(N, (int) blast_set.size());
        total_recall += query_recall;

        ofs << "Query: " << query_ids[q] << endl;
        ofs << "Query Recall: " << query_recall << endl;
        ofs << "Query Time: " << chrono::duration<double>(t1 - t0).count() << endl;
        for (int i = 0; i < approximate_neighbors.size(); i++) {
            int neighbor_idx = approximate_neighbors[i].image_id;
            string neighbor_id = protein_ids[neighbor_idx];

            // Get blast identity (0 if not in blast)
            double identity = 0.0;
            auto qit = blast_identities.find(q);

            if (qit != blast_identities.end()) {
                auto pit = qit->second.find(neighbor_idx);

                if (pit != qit->second.end())
                    identity = pit->second;
            }

            // Check if neighbor is in top N blast hits
            string inN = blast_set.count(neighbor_idx) ? "Yes" : "No";
            // Print neighbors
            ofs << "Nearest neighbor-" << (i + 1) << ": " << neighbor_id << endl;
            ofs << "distanceApproximate: " << approximate_neighbors[i].distance_to_neighbor << endl;
            ofs << "Blast Identity: " << identity << endl;
            ofs << "In BLAST top N: " << inN << endl;
        }
        ofs << "R-near neighbors:" << endl;
        // If there are range neighbors, print them
        if(!range_neighbors_ids.empty()) {
            for (int id : range_neighbors_ids) {
                ofs << id << endl;
            }
        }
    }
    ofs.close();

    // Calculate and store evaluation metrics for hypercube
    Metrics metrics;
    metrics.method = "Hypercube";
    metrics.average_AF = total_AF / queries.getRows();
    metrics.recall_at_N = total_recall / queries.getRows();
    metrics.qps = queries.getRows() / total_approx_time;
    metrics.average_approximate_time = total_approx_time / queries.getRows();
    metrics.average_true_time = total_true_time / queries.getRows();

    return metrics;
}


// Print results on output_file
void Hypercube::print_results(Metrics results, string& output_file) {
    ofstream ofs(output_file, ios::app);

    // Print the total metrics
    ofs << "Average AF: " << results.average_AF << endl;
    ofs << "Recall@N: " << results.recall_at_N << endl;
    ofs << "QPS: " << results.qps <<endl;
    ofs << "tApproximateAverage: " << results.average_approximate_time << endl;
    ofs << "tTrueAverage: " << results.average_true_time;

    ofs.close();
}
