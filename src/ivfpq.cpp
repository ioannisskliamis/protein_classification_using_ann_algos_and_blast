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
#include <tuple>
#include <chrono>

#include "ivfpq.hpp"

using namespace std;

inline vector<float> subtract(const vector<float>& a, const vector<float>& b) {                     //Function to find the residual
    assert(a.size() == b.size());

    size_t dimensionality = a.size();                                                               //Find dimensions of vectors
    vector<float> result(dimensionality);

    for(size_t i = 0; i < dimensionality; i++) {                                                    //Find the residual vector of a - b
        result[i] = a[i] - b[i];
    }

    return result;
}


IVFPQ::IVFPQ(string& input_stream, string& query_stream, const int& seed, const int& kclusters, const int& nprobe, const int& N, const int& R, 
			 const int& M, const int& cluster_nbits, const bool& range_search, double (*distance_fun)(const std::vector<float>&, const std::vector<float>&), 
			 const string& type) : IVFBase(input_stream, query_stream, seed, kclusters, nprobe, N, R, range_search, distance_fun, type), M(M), cluster_nbits(cluster_nbits) {}


tuple<IVFBase::clusteringResult, vector<IVFPQ::PQEntry>, vector<vector<vector<float>>>> IVFPQ::buildIVFPQ() {

    vector<Residuals> img_res(input.getRows());
    vector<Subvectors> imgs_subvecs(input.getRows());

    //Create the vector that will hold the residuals of all images
    for(int i = 0; i < input.getRows(); i++) {
        img_res[i].img_id = i;
        img_res[i].residual.resize(input.getColumns());
    }

    //Create the vector that will hold the subvectors of all images
    for(int i = 0; i < input.getRows(); i++) {
        imgs_subvecs[i].img_id = i;
        imgs_subvecs[i].vectors.resize(M);
    }

    clusteringResult result;
    result = LloydsAlgorithm(subset, k_clusters);                                     //Run Lloyds for subset to find centroids

    int dim = input.getColumns();
    int subdim = dim / M;

    for(int i = 0; i < result.clusters.size(); i++){		                          //Assign the correct id of the images from the subset

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
    for(int i = 0; i < input.getRows(); i++) {

    	//Check if image is part of the  subset
        if (is_subset[i])
            continue;

        //Find closest centroid
        double min_dist;
        int    min_center_index = -1;

        for(int j = 0; j < result.clusters.size(); j++) {		
            double curr_dist = distance_fn(input[i], result.clusters[j].centroid);

            if ((min_center_index == -1) || (curr_dist < min_dist)) {
                min_dist = curr_dist;
                min_center_index = j;
            }
        }

        assert(min_center_index != -1);
        result.clusters[min_center_index].img_ids.push_back(i);
    }

    //Find residuals of each image
    for(int i = 0; i < result.clusters.size(); i++) {

        for(int j = 0; j < result.clusters[i].img_ids.size(); j++) {
            img_res[result.clusters[i].img_ids[j]].residual = subtract(input[result.clusters[i].img_ids[j]], result.clusters[i].centroid);
            img_res[result.clusters[i].img_ids[j]].closest_centroid = i;
        }
    }

    //For each residual of image find it's subvectors
    for(int i = 0; i < img_res.size() ; i++) {

        for(int m = 0; m < M; m++) {
            int start = m * subdim;
            int end = (m == M - 1) ? dim : start + subdim;
            imgs_subvecs[i].vectors[m] = vector<float>(img_res[i].residual.begin() + start, img_res[i].residual.begin() + end);
        }

    }

    //vector<clusteringResult> m_results;
    vector<PQEntry> pqentries;                                                                      //PQ entries vector that holds the codes of all images
    pqentries.resize(input.getRows());

    for (int i = 0; i < input.getRows(); i++) {
        pqentries[i].img_id = i;
        pqentries[i].code.resize(M);
    }

    int subspace_clusters = 1 << cluster_nbits;

    vector<vector<vector<float>>> codebook(M, vector<vector<float>>(subspace_clusters, vector<float>(subdim))); //Codebook for centroids of subspaces

    //For each subspace do clustering
    for(int m = 0; m < M; m++) {
        cout << "Subspace " << m << endl;                                                           //I print this just to keep track

        vector<vector<float>> subvectors;

        //Take the m-th subvector of each image and create a Dataset
        for(int i = 0; i < imgs_subvecs.size(); i++) {
        	vector<float> subvector;
            subvector = imgs_subvecs[i].vectors[m];
            subvectors.push_back(subvector);
        }
        Dataset residual_set(subvectors, data_type);

        //Run LLoyd's for the m-th subvectors of each image
        clusteringResult m_result;
        m_result = LloydsAlgorithm(residual_set, subspace_clusters);
        //m_results.push_back(m_result);                                                            //We don't need to keep this

        //Create codebook and fill pqentries vector (the codes for each image)
        for(int i = 0; i < m_result.clusters.size(); i++){

            for(int j = 0; j < m_result.clusters[i].img_ids.size(); j++){
                pqentries[m_result.clusters[i].img_ids[j]].code[m] = i;
                assert(pqentries[m_result.clusters[i].img_ids[j]].img_id == m_result.clusters[i].img_ids[j]);   //Check if the id is indeed the correct one
            }
            codebook[m][i] = m_result.clusters[i].centroid;                                         //Fill the codebook with centroids values
        }

    }

    return make_tuple(result, pqentries, codebook);
}

Metrics IVFPQ::IVFPQQuery(const IVFBase::clusteringResult& clusteringResult, const vector<IVFPQ::PQEntry>& pqentries, const vector<vector<vector<float>>>& codebook, string& outstream) const {
	std::ofstream ofs(outstream);
	double total_approx_time = 0.0, total_true_time = 0.0, total_AF = 0.0, total_recall = 0.0;

    auto [protein_id_index, protein_ids] = readProteinIdMap("protein_ids.txt");
    auto [query_id_index, query_ids] = readQueryIdMap("targets.fasta");

    unordered_map<int, vector<int>> blast_hits = readTSV("blast_results.tsv", protein_id_index , query_id_index, neighbors);
    unordered_map<int,unordered_map<int, double>> blast_identities = readIdentities("blast_results.tsv", protein_id_index , query_id_index);

    for(int query_id = 0; query_id < query.getRows(); query_id++) {
    	auto t0 = chrono::system_clock::now();
        vector<CentroidDist> dist_to_centroids;

        vector<int> range_neighbors_ids;

        for(int j = 0; j < clusteringResult.clusters.size(); j++){
            double q_to_cen = (double) distance_fn(query[query_id], clusteringResult.clusters[j].centroid);

            dist_to_centroids.push_back({ q_to_cen, j });
        }

        sort(dist_to_centroids.begin(), dist_to_centroids.end(), [](const CentroidDist& a, const CentroidDist& b) {
            return a.distance < b.distance;
        });

        vector<int> closest_centroids;

        //Find coarsed centroids of query
        for(int i = 0; i < n_probe; i++)
            closest_centroids.push_back(dist_to_centroids[i].centroid_id);

        //Find residual of query
        vector<Neighbor> K_cand;

        for(int i = 0; i < closest_centroids.size(); i++) {
            //Residuals res;
            vector<float> residual(query.getColumns());
            residual = subtract(query[query_id], clusteringResult.clusters[closest_centroids[i]].centroid);

            int dim = input.getColumns();
            int subdim = dim / M;

            vector<vector<float>> r_sub;
            r_sub.resize(M);

            //Split residual to M parts
            for(int m = 0; m < M; m++) {
                int start = m * subdim;
                int end = (m == M - 1) ? dim : start + subdim;
                r_sub[m] = vector<float>(residual.begin() + start, residual.begin() + end);
            }

            int subspace_clusters = 1 << cluster_nbits;

        	vector<vector<double>> lut;
        	lut.resize(M);

            //Create lookup table of image
        	for(int m = 0; m < M; m++) {
        		lut[m].resize(subspace_clusters);

        		for(int i = 0; i < subspace_clusters; i++){
        			lut[m][i] = 0.0;
        		}
        	}

            for(int m = 0; m < M; m++) {

                for(int k = 0; k < subspace_clusters; k++) {
                    lut[m][k] = (double) distance_fn(r_sub[m], codebook[m][k]);
                }
            }

            //For nprobe clusters find candidate neighbors
            for(int j = 0; j < clusteringResult.clusters[closest_centroids[i]].img_ids.size(); j++) {
                Neighbor nei;
                PQEntry pqentry;
                pqentry = pqentries[clusteringResult.clusters[closest_centroids[i]].img_ids[j]];

                double approx_dist = 0.0;

                for(int m = 0; m < M; m++) {
                    approx_dist += lut[m][pqentry.code[m]];
                }

                nei.distance_to_neighbor = approx_dist;
                nei.image_id = clusteringResult.clusters[closest_centroids[i]].img_ids[j];

                if (range_search && approx_dist <= radius) {
            		range_neighbors_ids.push_back(nei.image_id);
            	}
                K_cand.push_back(nei);
            }
        }

        //Sort candidates
        sort(K_cand.begin(), K_cand.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.distance_to_neighbor < b.distance_to_neighbor;
        });

        vector<Neighbor> K_closest;

        for(int i = 0; i < neighbors && i < K_cand.size(); i++) {
            K_closest.push_back(K_cand[i]);
        }

        auto t1 = chrono::system_clock::now();

        total_approx_time += chrono::duration<double>(t1 - t0).count();

        // Perform exhaustive search
        vector<Neighbor> all_true_neighbors;
        vector<int> true_ids;
        vector<double> true_distances;
        
        auto t2 = chrono::system_clock::now();
        all_true_neighbors = input.getKNearestNeighbors(distance_fn, query[query_id], neighbors);

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
        auto& blast_ids = blast_hits[query_id];

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
        ofs << "Query: " << query_ids[query_id] << endl;
        ofs << "Query Recall: " << query_recall << endl;
        ofs << "Query Time: " << chrono::duration<double>(t1 - t0).count() << endl;
        for (int i = 0; i < K_closest.size(); i++) {
            int neighbor_idx = K_closest[i].image_id;
            string neighbor_id = protein_ids[neighbor_idx];

            // Get blast identity (0 if not in blast)
            double identity = 0.0;
            auto qit = blast_identities.find(query_id);

            if (qit != blast_identities.end()) {
                auto pit = qit->second.find(neighbor_idx);

                if (pit != qit->second.end())
                    identity = pit->second;
            }

            // Check if neighbor is in top N blast hits
            string inN = blast_set.count(neighbor_idx) ? "Yes" : "No";
            // Print neighbors
            ofs << "Nearest neighbor-" << (i + 1) << ": " << neighbor_id << endl;
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
    metrics.method = "IVFPQ";
    metrics.average_AF = total_AF / query.getRows();
    metrics.recall_at_N = total_recall /  query.getRows();
    metrics.qps = query.getRows() / total_approx_time;
    metrics.average_approximate_time = total_approx_time / query.getRows();
    metrics.average_true_time = total_true_time / query.getRows();

    return metrics;
}
