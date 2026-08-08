#include "readFuncs.hpp"

//Function that reads blast_results.tsv file and maps queries to blast hits protein indices
unordered_map<int, vector<int>> readTSV(const string& filename, const unordered_map<string,int>& protein_id_index, const unordered_map<string,int>& query_id_index, int N) {
    unordered_map<int, vector<int>> blast_hits;
    ifstream tsv(filename);
    double threshold = 0.01;

    if (!tsv.is_open()) 
        throw runtime_error("Cannot open BLAST TSV file.");

    string line;

    while (getline(tsv, line)) {

        if (line.empty()) 
        continue;

        stringstream ss(line);
        string qseqid, sseqid, dummy;
        double evalue;

        // Parse blast tabular format
        ss >> qseqid >> sseqid >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> evalue;

        // Ignore high e-value hits
        if (evalue > threshold) 
            continue; 

        // Lookup indices
        auto qit = query_id_index.find(qseqid);
        if (qit == query_id_index.end())
            continue;

        auto pit = protein_id_index.find(sseqid);
        if (pit == protein_id_index.end())
        	continue;

        int query_idx = qit->second;
        int protein_idx = pit->second;

        // Store up to N hits per query
        if ((int) blast_hits[query_idx].size() < N)
            blast_hits[query_idx].push_back(protein_idx);
    }

    return blast_hits;
}

//Function that reads porotein_ids.txt file and maps proteins ids to indices in input dataset
pair<unordered_map<string,int>, vector<string>> readProteinIdMap(const string& filename) {
    unordered_map<string,int> protein_id_index;
    vector<string> protein_ids;
    ifstream in(filename);
    string line;
    int idx = 0;

    while (std::getline(in,line)) {                                     // Read line in file

        if (!line.empty()) {
            protein_id_index[line] = idx;                               // Map to correct index of dataset
            idx++;
            protein_ids.push_back(line);
        }

    }

    return { protein_id_index, protein_ids };
}

//Function that reads targets.fasta file and maps query proteins ids to indices in query dataset
pair<unordered_map<string,int>, vector<string>> readQueryIdMap(const string& query_fasta) {
    unordered_map<string,int> query_id_index;
    vector<string> query_ids;
    ifstream ifs(query_fasta);

    string line;
    int idx = 0;

    while (getline(ifs, line)) {                                        // Read line in file

        if (line.empty()) 
            continue;

        if (line[0] == '>') {                                           // Check if it starts with '>' symbol
            string id = line.substr(1);                                 // Remove '>'
            query_id_index[id] = idx;                                   // Map to correct index
            idx++;
            query_ids.push_back(id);
        }

    }

    return { query_id_index, query_ids };
}

//Function that reads blas_results.tsv file and maps queries to blast hits to blast hits and their identity
unordered_map<int, unordered_map<int,double>> readIdentities(const string& filename, const unordered_map<string,int>& protein_id_index, const unordered_map<string,int>& query_id_index) {
    unordered_map<int, unordered_map<int,double>> blast_identities;
    ifstream tsv(filename);

    if (!tsv.is_open()) 
        throw runtime_error("Cannot open BLAST TSV file.");

    string line;

    while (getline(tsv, line)) {

        if (line.empty()) 
        continue;

        stringstream ss(line);
        string qseqid, sseqid, dummy;
        double evalue, pident;

        ss >> qseqid >> sseqid >> pident >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> evalue;

        auto qit = query_id_index.find(qseqid);
        if (qit == query_id_index.end())
            continue;

        auto pit = protein_id_index.find(sseqid);
        if (pit == protein_id_index.end())
            continue;

        int query_idx = qit->second;
        int protein_idx = pit->second;
        
        blast_identities[query_idx][protein_idx] = pident;
    }

    return blast_identities;
}