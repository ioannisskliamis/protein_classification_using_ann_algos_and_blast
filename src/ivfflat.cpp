#include <algorithm>
#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include <numeric>
#include <unordered_set>
#include <chrono>
#include <queue>


#include "ivfflat.hpp"

using namespace std;

IVFFlat::IVFFlat(string& input_stream, string& query_stream, const int& seed, const int& kclusters,
            	 const int& nprobe, const int& N, const int& R, const bool& range_search, double (*distance_fn)(const std::vector<float>&, const std::vector<float>&), 
            	 const string& type) : IVFBase(input_stream, query_stream, seed, kclusters, nprobe, N, R, range_search, distance_fn, type) {}
			

IVFBase::clusteringResult IVFFlat::buildIVFFlat() {
    bool flag = false;

    clusteringResult result;
    result = LloydsAlgorithm(subset, k_clusters);                                       //Run Lloyds for subset to find centroids

    for(int i = 0; i < result.clusters.size();i++){
        
        for(int j = 0; j < (int) result.clusters[i].img_ids.size(); j++) {
            int subset_local_id = result.clusters[i].img_ids[j];
            assert(subset_local_id >= 0 && subset_local_id < subset_indices.size());
            result.clusters[i].img_ids[j] = subset_indices[subset_local_id];
        }
    }

    vector<char> is_subset(input.getRows(), 0);

    for(int idx : subset_indices)
        is_subset[idx] = 1;

    //Assign rest of the images to centroids
    for(int i = 0; i < input.getRows(); ++i) {                                          //Assign rest of the images to centroids

        //Check if image is part of the  subset
        if (is_subset[i])
            continue;

        double min_dist;
        int    min_center_index = -1;

        for(int j = 0; j < k_clusters; j++) {
            double curr_dist = distance_fn(input[i], result.clusters[j].centroid);

            if ((min_center_index == -1) || (curr_dist < min_dist)) {
                min_dist = curr_dist;
                min_center_index = j;
            }
        }

        assert(min_center_index != -1);
        result.clusters[min_center_index].img_ids.push_back(i);
    }

    return result;
}

Metrics IVFFlat::IVFFlatQuery(const IVFBase::clusteringResult& clusteringResult, string& outstream) const{
    vector<Results> results;
	double total_approx_time = 0.0, total_true_time = 0.0, total_AF = 0.0, total_recall = 0.0;

    auto [protein_id_index, protein_ids] = readProteinIdMap("protein_ids.txt");
    auto [query_id_index, query_ids] = readQueryIdMap("targets.fasta");

    unordered_map<int, vector<int>> blast_hits = readTSV("blast_results.tsv", protein_id_index , query_id_index, neighbors);
    unordered_map<int,unordered_map<int, double>> blast_identities = readIdentities("blast_results.tsv", protein_id_index , query_id_index);

    std::ofstream ofs(outstream);

    for(int i = 0; i < query.getRows(); i++) {

    	auto t0 = chrono::system_clock::now();

        vector<CentroidDist> dist_to_centroids;

        vector<int> range_neighbors_ids;

        //Find closest nprone clusters to query
        for(int j = 0; j < clusteringResult.clusters.size(); j++){
            double q_to_cen = distance_fn(query[i], clusteringResult.clusters[j].centroid);

            dist_to_centroids.push_back({ q_to_cen, j });
        }

        sort(dist_to_centroids.begin(), dist_to_centroids.end(), [](const CentroidDist& a, const CentroidDist& b) {
            return a.distance < b.distance;
        });

        vector<int> closest_centroids;

        for(int n = 0; n < n_probe; n++)
            closest_centroids.push_back(dist_to_centroids[n].centroid_id);

        vector<Neighbor> cand;

        //For each of the nprobe clusters find the distance to images of that cluster
        for(int cent_id : closest_centroids) {
            for(int j = 0; j < clusteringResult.clusters[cent_id].img_ids.size(); j++) {
                double dist = (double) distance_fn(query[i], input[clusteringResult.clusters[cent_id].img_ids[j]]);
                int neighbor_id = clusteringResult.clusters[cent_id].img_ids[j];

                if (range_search && dist <= radius) {
            		range_neighbors_ids.push_back(neighbor_id);
            	}
                cand.push_back({ dist, neighbor_id });
            }
        }
        
        //Sort candidates
        sort(cand.begin(), cand.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.distance_to_neighbor < b.distance_to_neighbor;
        });

        vector<Neighbor> K_closest;

        for(int i = 0; i < neighbors && i < cand.size(); i++)
            K_closest.push_back(cand[i]);

        auto t1 = chrono::system_clock::now();
        total_approx_time += chrono::duration<double>(t1 - t0).count();

        // Perform exhaustive search
        vector<Neighbor> all_true_neighbors;
        vector<int> true_ids;
        vector<double> true_distances;
        
        auto t2 = chrono::system_clock::now();
        all_true_neighbors = input.getKNearestNeighbors(distance_fn, query[i], neighbors);

        // Retrieve ids and distances of N nearest neighbors
        for (int i = 0; i < all_true_neighbors.size(); i++) {
            true_distances.push_back(all_true_neighbors[i].distance_to_neighbor);
            true_ids.push_back(all_true_neighbors[i].image_id);
        }

        // True distance (exhaustive search) timer
        auto t3 = chrono::system_clock::now();
        total_true_time += chrono::duration<double>(t3 - t2).count();

        // Evaluation of current query
        if(!K_closest.empty() && !true_distances.empty()) {
            if (true_distances[0] > 0)
                total_AF += K_closest[0].distance_to_neighbor / true_distances[0]; // Approximation factor
            else
                total_AF += 1.0;
        }

        // Find true neighbors among approximates
        auto& blast_ids = blast_hits[i];

        if (blast_ids.empty()) {
            total_recall += 0.0;
        }

        unordered_set<int> blast_set(blast_ids.begin(), blast_ids.begin() + min(neighbors, (int)blast_ids.size()));     

        int hits_per_query = 0;
        
        for(int k = 0; k < min(neighbors, (int)K_closest.size()); k++) {
            if (blast_set.count(K_closest[k].image_id)) {
                hits_per_query++;
            }
        }

        double query_recall = (double) hits_per_query / min(neighbors, (int) blast_set.size());
        
        total_recall += query_recall;

        // Print results
        ofs << "Query: " << query_ids[i] << endl;
        ofs << "Query Recall: " << query_recall << endl;
        ofs << "Query Time: " << chrono::duration<double>(t1 - t0).count() << endl;
        for (int j = 0; j < K_closest.size(); j++) {
            int neighbor_idx = K_closest[j].image_id;
            string neighbor_id = protein_ids[neighbor_idx];

            // Get blast identity (0 if not in blast)
            double identity = 0.0;
            auto qit = blast_identities.find(i);

            if (qit != blast_identities.end()) {
                auto pit = qit->second.find(neighbor_idx);

                if (pit != qit->second.end())
                    identity = pit->second;
            }

            // Check if neighbor is in top N blast hits
            string inN = blast_set.count(neighbor_idx) ? "Yes" : "No";
            // Print neighbors
            ofs << "Nearest neighbor-" << (j + 1) << ": " << neighbor_id << endl;
            ofs << "distanceApproximate: " << K_closest[i].distance_to_neighbor << endl;
            ofs << "Blast Identity: " << identity << endl;
            ofs << "In BLAST top N: " << inN << endl;
        }
        if (!range_neighbors_ids.empty()) {
            ofs << "R-near neighbors:" << endl;
            for (int id : range_neighbors_ids) {
                ofs << id << endl;
            }
        }
        
    }

    ofs.close();

    Metrics metrics;
    metrics.method = "IVFFlat";
    metrics.average_AF = total_AF / query.getRows();
    metrics.recall_at_N = total_recall / query.getRows();
    metrics.qps = query.getRows() / total_approx_time;
    metrics.average_approximate_time = total_approx_time / query.getRows();
    metrics.average_true_time = total_true_time / query.getRows();

    return  metrics;
}

void IVFFlat::IVFFlatQueryGraph(const IVFBase::clusteringResult& clusteringResult, string& outstream) const{
    std::ofstream ofs(outstream);

    for(int i = 0; i < query.getRows(); i++) {

        vector<CentroidDist> dist_to_centroids;

        vector<int> range_neighbors_ids;

        //Find closest nprone clusters to query
        for(int j = 0; j < clusteringResult.clusters.size(); j++){
            double q_to_cen = distance_fn(query[i], clusteringResult.clusters[j].centroid);

            dist_to_centroids.push_back({ q_to_cen, j });
        }

        sort(dist_to_centroids.begin(), dist_to_centroids.end(), [](const CentroidDist& a, const CentroidDist& b) {
            return a.distance < b.distance;
        });

        vector<int> closest_centroids;

        for(int n = 0; n < n_probe; n++)
            closest_centroids.push_back(dist_to_centroids[n].centroid_id);

        vector<Neighbor> cand;

        //For each of the nprobe clusters find the distance to images of that cluster
        for(int cent_id : closest_centroids) {
            for(int j = 0; j < clusteringResult.clusters[cent_id].img_ids.size(); j++) {
                double dist = (double) distance_fn(query[i], input[clusteringResult.clusters[cent_id].img_ids[j]]);
                int neighbor_id = clusteringResult.clusters[cent_id].img_ids[j];

                if (range_search && dist <= radius) {
            		range_neighbors_ids.push_back(neighbor_id);
            	}
                // Exclude himself
                if(dist > 0) {
                    cand.push_back({ dist, neighbor_id });
                }
            }
        }

        priority_queue<Neighbor> neighbor_pq;

        for(auto &nb : cand) {
            if (neighbor_pq.size() < neighbors)
                neighbor_pq.push(nb);
            else if (nb.distance_to_neighbor < neighbor_pq.top().distance_to_neighbor) {
                neighbor_pq.pop();
                neighbor_pq.push(nb);
            }
        }

        vector<Neighbor> K_closest;
        K_closest.reserve(neighbor_pq.size());

        while (!neighbor_pq.empty()) {
            K_closest.push_back(neighbor_pq.top());
            neighbor_pq.pop();
        }

        reverse(K_closest.begin(), K_closest.end());


        // Print results
        ofs << "Query: " << i << endl;
        for (int i = 0; i < K_closest.size(); i++) {
            ofs << "Nearest neighbor-" << (i + 1) << ": " << K_closest[i].image_id << endl;
            ofs << "distanceApproximate: " << K_closest[i].distance_to_neighbor << endl;
        }

    }

    ofs.close();
}
