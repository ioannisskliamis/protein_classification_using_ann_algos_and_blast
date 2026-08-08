#include "LSH.hpp"

// Constructor
LSH::LSH(string& input, string& query, const int& seed, const int& k,
                const int& L, const double& w, const int& N, const double& R, const bool& range, 
                double (*distance_fun)(const vector<float>&, const vector<float>&), string& type) : data(input, type),
                queries(query, type), seed(seed), k(k), L(L), w(w), N(N), R(R), range_search(range), distance_fn(distance_fun), data_type(type) {}


// Creates L group of k hash functions
void LSH::initializeHashFunctions() {
    // Take the dimensions of a vector
    size_t vector_dim = data.getColumns();
    // Initialize random generator
    normal_distribution<double> normal(0.0, 1.0);
    uniform_real_distribution<double> uniform_w(0.0, 1.0);
    // For the hash function
    int M = UINT_MAX - 5;
    r.resize(k);
    uniform_int_distribution<int> uniform_r(1, M - 1);
    for (int i = 0; i < k; i++) {
        r[i] = uniform_r(rng);
    }

    hash_functions.resize(L, vector<RandomProjection>(k));
    // Create L groups of k hash functions
    for (int i = 0; i < L; i++) { // For each hash table
        for (int hi = 0; hi < k; hi++) { // Repeat k times to form g
            RandomProjection hash_func_i;
            hash_func_i.p.resize(vector_dim); // Resize vector to create slots in advance
            
            // Get random projections for the vector's dimensions
            for (size_t dim = 0; dim < vector_dim; dim++) {
                hash_func_i.p[dim] = normal(rng);
            }
            
            // Get random t to shift
            hash_func_i.t = uniform_w(rng) * w;
            
            // Store
            hash_functions[i][hi] = hash_func_i;
        }
    }
}


// Insert vectors in L hash tables
void LSH::buildHashTables() {
    // Take the dimensions of a vector
    size_t vector_dim = data.getColumns();
    // For the hash function
    int M = UINT_MAX - 5;
    hash_tables.resize(L);
    // Insert image vectors in L hash tables using the modular combination from lecture slides
    for (int idx = 0; idx < data.getRows(); idx++) {
        for (int i = 0; i < L; i++) {
            // Vector to store k individual hash values
            vector<int> hvals(k);
            for (int hi = 0; hi < k; hi++) {
                // Inner product for projection (p * v)
                double inner_product = 0.0;
                for (size_t d = 0; d < vector_dim; d++) {
                    inner_product += hash_functions[i][hi].p[d] * data[idx][d];
                }
                // LSH formula: h(p) = (p * v + t)/w
                double hp = (inner_product + hash_functions[i][hi].t) / w;

                hvals[hi] = (int)floor(hp);
            }
            int IDp = 0;
            for (int hi = 0; hi < k; hi++) {
                IDp = (IDp + ((r[hi] * (int)(hvals[hi] % M)) % M)) % M; // IDp formula for quering trick
            }
            int g = IDp % (data.getRows() / 4); //g(p) = IDp mod TableSize

            hash_tables[i][g].push_back({idx, IDp});
        }
    }
}


// Performs ANN
Metrics LSH::search(string& output_file) {
    ofstream ofs(output_file);
    // Take the dimensions of a vector
    size_t vector_dim = data.getColumns();
    // For the hash function
    int M = UINT_MAX - 5;

    // Metrics
    double total_approx_time = 0.0, total_true_time = 0.0, total_AF = 0.0, total_recall = 0.0;

    auto [protein_id_index, protein_ids] = readProteinIdMap("protein_ids.txt");
    auto [query_id_index, query_ids] = readQueryIdMap("targets.fasta");

    unordered_map<int, vector<int>> blast_hits = readTSV("blast_results.tsv", protein_id_index , query_id_index, N);
    unordered_map<int,unordered_map<int, double>> blast_identities = readIdentities("blast_results.tsv", protein_id_index , query_id_index);

    // For each query of the dataset
    for (int q = 0; q < queries.getRows(); q++) {
        auto query = queries[q];
        
        auto t0 = chrono::system_clock::now();
        
        // Stores unique indices of data points that are candidate neighbors
        vector<int> candidates;
        vector<int> range_neighbors_ids;

        for (int i = 0; i < L; i++) {
            // Combined hash value for current query in current hash table
            int IDq = 0;
            vector<int> hvals(k);

            for (int hi = 0; hi < k; hi++) {
                // Inner product for projection (p * v)
                double inner_product = 0.0;
                for (int d = 0; d < vector_dim; d++) {
                    inner_product += hash_functions[i][hi].p[d] * query[d];
                }
                // LSH formula: h(p) = (p * v + t)/w
                double hp = (inner_product + hash_functions[i][hi].t) / w;
                hvals[hi] = (int)floor(hp);
                IDq = (IDq + ((r[hi] * (int)(hvals[hi] % M)) % M)) % M; // Compound hash key
            }
            int g = IDq % (data.getRows() / 4); //g(p) = IDp mod TableSize

            // Lookup "g" in the current hash table
            auto it = hash_tables[i].find(g);
            if(it != hash_tables[i].end()) {
                // Search data points stored in bucket "g"
                for (auto& entry : it->second) {
                    if(entry.ID == IDq) {
                        // If bucket "g" and the compound IDq match then then point is a candidate neighbor
                        candidates.push_back(entry.idx);
                    }
                }
            }
        }

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

        // Approximate distance timer
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

    // Calculate and store evaluation metrics for lsh
    Metrics metrics;
    metrics.method = "LSH";
    metrics.average_AF = total_AF / queries.getRows();
    metrics.recall_at_N = total_recall / queries.getRows();
    metrics.qps = queries.getRows() / total_approx_time;
    metrics.average_approximate_time = total_approx_time / queries.getRows();
    metrics.average_true_time = total_true_time / queries.getRows();

    return metrics;
    
}

// Print results on output file
void LSH::print_results(Metrics results, string& output_file) {
    // Print results on output_file
    ofstream ofs(output_file, ios::app);

    // Print the total metrics
    ofs << "Average AF: " << results.average_AF << endl;
    ofs << "Recall@N: " << results.recall_at_N << endl;
    ofs << "QPS: " << results.qps <<endl;
    ofs << "tApproximateAverage: " << results.average_approximate_time << endl;
    ofs << "tTrueAverage: " << results.average_true_time;

    ofs.close();
}