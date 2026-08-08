#ifndef __PARSINGFUNCS__
#define __PARSINGFUNCS__

#include <unordered_map>

using namespace std;

// Argument validation functions
bool is_int(const string& s);
bool is_double(const string& s);
bool is_boolean(const string& s);

// Parse integer argument (if not given then use default parameter)
int get_int_arg(unordered_map<string, string>& args, string flag, int default_val);

// Parse double argument (if not given then use default parameter)
double get_double_arg(unordered_map<string, string>& args, string flag, double default_val);

// Parse string argument (if not given then use default parameter)
string get_string_arg(unordered_map<string, string>& args, string flag, string default_val);

// Parse boolean argument (if not given then use default parameter)
bool get_boolean_arg(unordered_map<string, string>& args, string flag, bool default_val);

#endif // __PARSINGFUNCS__