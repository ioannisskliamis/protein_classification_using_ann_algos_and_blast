#include <iostream>
#include <string>
#include <unordered_map>

#include "LSH.hpp"
#include "Hypercube.hpp"
#include "euclidean.hpp"
#include "parsingFuncs.hpp"
#include "ivfbase.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    try {
        unordered_map<string, string> args;
        int alg_cnt = 0;

        // Parse cmd line arguments
        for (int i = 0; i < argc; i++) {
            string key = argv[i];
            if(key[0] == '-') { // Check if current arg is a flag
                //Check if the next argument is a value
                if(i + 1 < argc && argv[i+1][0] != '-') {
                    args[key] = argv[++i]; // Assign the value and skip the next argument
                } else {
                    if(key == "-lsh" || key == "-hypercube" || key == "-ivfflat" || key == "-ivfpq") {
                        args[key] = "true"; // If the flag is for algorithm just tag it
                        alg_cnt++;
                    } else {
                        args[key] = ""; // Mark as missing argument
                    }
                }
            }
        }

        if(alg_cnt > 1) {
            throw runtime_error("Multiple algorithm flags specified. Please choose one of: [-lsh | -hypercube | -ivfflat | -ivfpq]");
        }

        // Enforce mandatory arguments
        if(args.find("-d") == args.end() || args.find("-q") == args.end() || args.find("-o") == args.end() || args.find("-type") == args.end()) {
            throw invalid_argument("Missing required arguments: -d <input file> -q <query file> -o <output file> -type <mnist/sift/numpy>");
        }

        // Parse common arguments
        string input_file = get_string_arg(args, "-d", "");
        if(input_file.empty()) {
            throw runtime_error("Missing -d argument");
        }
        string query_file = get_string_arg(args, "-q", "");
        if(query_file.empty()) {
            throw runtime_error("Missing -q argument");
        }
        string output_file = get_string_arg(args, "-o", "");
        if(output_file.empty()) {
            throw runtime_error("Missing -o argument");
        }
        string data_type = get_string_arg(args, "-type", "");
        if(data_type.empty()) {
            throw runtime_error("Missing -type argument");
        }
        int seed = get_int_arg(args, "-seed", 1);
        int N = get_int_arg(args, "-N", 1);
        double R;
        if(data_type == "mnist") {
            R = get_double_arg(args, "-R", 2000.0);
        }
        else if(data_type == "sift"){
            R = get_double_arg(args, "-R", 2.0);
        }
        else if(data_type == "protein");
        else {
            throw runtime_error("Invalid -type argument. Must be \"mnist\", \"sift\" or \"protein\"");
        }
        bool range_search = get_boolean_arg(args, "-range", true);


        // Parse selected algorithm
        if(args.find("-lsh") != args.end()) {
            // Parse LSH specific arguments
            int k = get_int_arg(args, "-k", 4);
            int L = get_int_arg(args, "-L", 5);
            double w = get_double_arg(args, "-w", 4.0);

            LSH lsh_obj(input_file, query_file, seed, k, L, w, N, R, range_search, EuclideanDistance, data_type);

            // Run LSH algorithm
            lsh_obj.solve(output_file);
        }
        else if(args.find("-hypercube") != args.end()) {
            // Parse Hypercube specific arguments
            int kproj = get_int_arg(args, "-kproj", 14);
            if (kproj > 64) {
                throw runtime_error("Please select -kproj in range [0-64]"); // Later on uint64 is used for bitmask
            }
            double w = get_double_arg(args, "-w", 4.0);
            int M = get_int_arg(args, "-M", 10);
            int probes = get_int_arg(args, "-probes", 2);

            Hypercube hypercube_obj(input_file, query_file, seed, kproj, w, M, probes, N, R, range_search, EuclideanDistance, data_type);

            // Run Hypercube algorithm
            hypercube_obj.solve(output_file);
        }
        else if(args.find("-ivfflat") != args.end()) {
            if(args.find("-graph") != args.end()) {
                // Parse IVFFLAT specific arguments
                int kclusters = get_int_arg(args, "-kclusters", 50);
                int nprobe = get_int_arg(args, "-nprobe", 5);

                IVFFlat ivfflat_obj(input_file, query_file, seed, kclusters, nprobe, N, R, range_search, EuclideanDistance, data_type);

                // Run IVFLAT algorithm
                ivfflat_obj.create_graph(output_file);
            } else {
                // Parse IVFFLAT specific arguments
                int kclusters = get_int_arg(args, "-kclusters", 50);
                int nprobe = get_int_arg(args, "-nprobe", 5);

                IVFFlat ivfflat_obj(input_file, query_file, seed, kclusters, nprobe, N, R, range_search, EuclideanDistance, data_type);

                // Run IVFLAT algorithm
                ivfflat_obj.solve(output_file);
            }

        }
        else if(args.find("-ivfpq") != args.end()) {
            // Parse IVFPQ specific arguments
            int kclusters = get_int_arg(args, "-kclusters", 50);
            int nprobe = get_int_arg(args, "-nprobe", 5);
            int M = get_int_arg(args, "-M", 16);
            int nbits = get_int_arg(args, "-nbits", 8);

            IVFPQ ivfpq_obj(input_file, query_file, seed, kclusters, nprobe, N, R, M, nbits, range_search, EuclideanDistance, data_type);
            
            // Run IVFPQ algorithm
            ivfpq_obj.solve(output_file);
        }
        else {
            throw invalid_argument("Wrong algorithm given. Select one from [-lsh, -hypercube, -ivfflat, ivfpq]");
        }


    } catch (exception& exc) {
        cerr << "Error: " << exc.what() << endl;
        cerr << "Default usage: ./search -d <input file> -q <query file> -o <output file> -type <mnist/sift> [-lsh | -hypercube | -ivfflat | -ivfpq]" << endl;
        return 1;
    }
    


    return 0;
}