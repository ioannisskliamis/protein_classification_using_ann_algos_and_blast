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

#include "ivfbase.hpp"

#define MAX_ITERATIONS 50

using namespace std;

inline double findMean(const vector<float>& values) {                                   //Function to find mean of values (we use it to update the centroids)

    if (values.empty())
        return 0.0;

    double sum = 0.0;

    for(int i = 0; i < values.size(); i++){
        sum += (double) (values[i]);
    }

    return sum / ((double) values.size());
}


IVFBase::IVFBase(string& input_stream, string& query_stream, const int& seed, const int& kclusters,
                const int& nprobe, const int& N, const int& R, const bool& range_search, 
                double (*distance_fun)(const std::vector<float>&, const std::vector<float>&), 
                const string& type) : input(input_stream, type), query(query_stream, type), seed(seed), k_clusters(kclusters), n_probe(nprobe), 
				neighbors(N), radius(R), range_search(range_search), distance_fn(distance_fun), data_type(type) {

        auto result = input.createSubset(seed);
        this->subset = result.first;
        this->subset_indices = result.second;
}


vector<int> IVFBase::centerInitialization(const Dataset& input, const int& k_clusters) const{
    std::mt19937                gen(seed); //Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<>  dist(0, input.getRows() - 1); // because s_values are integers

    vector<int> centers;

    centers.push_back(dist(gen));

    for(int i = 1; i < k_clusters; i++) {
        vector<int>    cand_center_ids;
        vector<double> partial_sums;
        partial_sums.push_back(0.0);
        
        //Finding the minimum distance between the current point
        //and a centroid
        for(int j = 0; j < input.getRows(); j++) {
            double min_dist     = -1.0;
            bool   min_dist_set = false;

            for(int k = 0; k < centers.size(); k++) {

                if (j == centers[k]) {
                    min_dist_set = false; 
                    break;
                }

                double cand_dist = distance_fn(input[j], input[centers[k]]);

                if ((!min_dist_set) || (cand_dist < min_dist)) {
                    min_dist     = cand_dist;
                    min_dist_set = true;
                }
            }

            if (!min_dist_set) 
                continue;

            //Creating the probability distribution
            partial_sums.push_back(partial_sums.back() + (min_dist * min_dist));
            cand_center_ids.push_back(j);
        }

        //Finding the new centroid by generating a real number and 
        //checking to what point's slot that number is present
        uniform_real_distribution<> real_dist(0, partial_sums.back());

        double real_dist_val = real_dist(gen);

        for(int j = 0; j < partial_sums.size() - 1; j++) {

            if ((partial_sums[j] < real_dist_val) && (real_dist_val <= partial_sums[j+1])) {
                centers.push_back(cand_center_ids[j]);
            }
        }

        assert(centers.size() == i + 1);
    }

    return centers;
}

// Classic Lloyd's algorithm
IVFBase::clusteringResult IVFBase::LloydsAlgorithm(const Dataset& input, const int& k_clusters) const{
    clock_t start, end;

    start = clock();

    // Initializing the centers
    vector<int> curr_centers = centerInitialization(input, k_clusters);
    cout << "Centers Initialized" << endl;
    clusteringResult  result;

    for(int i = 0; i < curr_centers.size(); i++) {
        result.clusters.push_back(clusteringResult::cluster(input[curr_centers[i]]));
    }

    bool to_stop = false;
    int curr_iterations = 0;

    while ((!to_stop) && (curr_iterations < MAX_ITERATIONS)) {
        curr_iterations++;

        cout << "Started iteration " << curr_iterations <<endl;    //I print that just for keep track of progress
        to_stop = true;

        assert(result.clusters.size() == k_clusters);

        //Assignment of vectors to centers
        for(int i = 0; i < input.getRows(); i++) {
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

        assert(result.clusters.size() == k_clusters);

        //Updating the centers
        for(int i = 0; i < k_clusters; i++) {
            vector<float> old_centroid = result.clusters[i].centroid;
            vector<float> new_centroid;

            for(int j = 0; j < old_centroid.size(); j++) {
                vector<float> j_th_dims;

                for(int k = 0; k < result.clusters[i].img_ids.size(); k++) {
                    j_th_dims.push_back(input[result.clusters[i].img_ids[k]][j]);
                }

                new_centroid.push_back((float) findMean(j_th_dims));
            }

            if (new_centroid != old_centroid) {
                result.clusters[i].centroid = new_centroid;
                to_stop = false;
            }
        }

        if ((!to_stop) && (curr_iterations < MAX_ITERATIONS)) {
            for(int i = 0; i < result.clusters.size(); i++)
                result.clusters[i].img_ids.clear();
        }
    }
    end = clock();

    result.cluster_time = ((double) end - start) / ((double) CLOCKS_PER_SEC);
    
    return result;
}

void IVFBase::clusteringResult::find_silhouettes(const Dataset& input, double (*distance_fn)(const vector<float>&, const vector<float>&)) {
    double total_silhouettes_sum = 0.0;
    int    total_silhouettes_num = 0;

    vector<int> all_ids(input.getRows());

    for(int i = 0; i < clusters.size(); i++) {
        double cluster_silhouettes_sum = 0.0;
        int    cluster_silhouettes_num = 0;

        if (clusters[i].img_ids.size() == 0) {
            silhouettes.push_back(0.0);
            continue;
        }

        for(int j = 0; j < clusters[i].img_ids.size(); j++){
            //Finding average distance with points in same cluster
            vector<float> curr_img_vector = input[ clusters[i].img_ids[j] ];
            double distance_sum = 0.0;
            int    distances    = 0;

            for(int k = 0; k < clusters[i].img_ids.size(); k++) {

                distance_sum += distance_fn(curr_img_vector, input[ clusters[i].img_ids[k] ]);
                
                distances++;
            }

            double clust_avg_dist = distance_sum / ((double) distances); // For point p this is a(p)

            //Finding index of 2nd nearest cluster
            int    nearest_clust_index = -1;
            double nearest_clust_dist  = 0;

            for(int k = 0; k < clusters.size(); k++) {

                if (k == i) 
                    continue;

                double cand_dist = distance_fn(curr_img_vector, clusters[k].centroid);

                if ((nearest_clust_index == -1) || (cand_dist < nearest_clust_dist)) {
                    nearest_clust_dist  = cand_dist;
                    nearest_clust_index = k; 
                }
            }

            //Finding average distace from points of 2nd nearest cluster
            distance_sum = 0.0;
            distances    = 0;

            for(int k = 0; k < clusters[nearest_clust_index].img_ids.size(); k++) {
                distance_sum += distance_fn(curr_img_vector, input[ clusters[nearest_clust_index].img_ids[k] ]);
                
                distances++;
            }

            double nearest_clust_avg_dist; // For point p this is b(p)

            if (distances == 0) {  // Empty cluster
                nearest_clust_avg_dist = 0.0;
            } else {
                nearest_clust_avg_dist = distance_sum / ((double) distances);
            }

            //Finding Silhouette
            double curr_silhouette = 0.0;

            if (clust_avg_dist < nearest_clust_avg_dist) {
                curr_silhouette = 1.0 - (clust_avg_dist / nearest_clust_avg_dist);
            } else if (clust_avg_dist > nearest_clust_avg_dist) {
                curr_silhouette = (nearest_clust_avg_dist / clust_avg_dist) - 1.0;
            }

            assert((curr_silhouette >= -1.0) && (curr_silhouette <= 1.0));

            cluster_silhouettes_sum += curr_silhouette;
            cluster_silhouettes_num++;
        }

        total_silhouettes_sum += cluster_silhouettes_sum;
        total_silhouettes_num += cluster_silhouettes_num;

        silhouettes.push_back(cluster_silhouettes_sum / ((double) cluster_silhouettes_num));
    }

    silhouettes.push_back(total_silhouettes_sum / ((double) total_silhouettes_num));
}

void IVFBase::print_results(Metrics results, string& outstream) {
    // Print results on output_file
    std::ofstream ofs(outstream, std::ios::app);

    ofs << "Average AF: " << results.average_AF << endl;
    ofs << "Recall@N: " << results.recall_at_N << endl;
    ofs << "QPS: " << results.qps <<endl;
    ofs << "tApproximateAverage: " << results.average_approximate_time << endl;
    ofs << "tTrueAverage: " << results.average_true_time;

    ofs.close();
}

void IVFBase::print_silhouettes(string& outstream) {

    clusteringResult result;
    result = LloydsAlgorithm(subset, k_clusters);       //Run Lloyds for subset to find centroids


    result.find_silhouettes(subset, distance_fn);

    std::ofstream ofs(outstream);

    ofs << "Silhouette: " << "[";

    for(int i = 0; i < result.silhouettes.size() - 1; i++) {
        ofs << result.silhouettes[i] << ", ";
    }

    ofs << result.silhouettes.back() << "]" << endl;


    ofs << "Total silhouettes score: ";
    ofs << result.silhouettes[result.silhouettes.size() - 1] << endl;

    ofs.close();
}
