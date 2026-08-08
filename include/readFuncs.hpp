#ifndef _READFUNCS_
#define _READFUNCS_

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>

using namespace std;

//Function that reads blast_results.tsv file and maps queries to blast hits protein indices
unordered_map<int, vector<int>> readTSV(const string& filename, const unordered_map<string,int>&, const unordered_map<string,int>&, int);

//Function that reads porotein_ids.txt file and maps proteins ids to indices in input dataset
pair<unordered_map<string,int>, vector<string>> readProteinIdMap(const string&);

//Function that reads targets.fasta file and maps query proteins ids to indices in query dataset
pair<unordered_map<string,int>, vector<string>> readQueryIdMap(const string&);

//Function that reads blas_results.tsv file and maps queries to blast hits to blast hits and their identity
unordered_map<int, unordered_map<int,double>> readIdentities(const string&, const unordered_map<string,int>& , const unordered_map<string,int>&);

#endif