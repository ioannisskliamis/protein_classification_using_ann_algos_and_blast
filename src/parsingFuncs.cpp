#include "parsingFuncs.hpp"

// Validate integer argument
bool is_int(const string& s) {
    if(s.empty()) return false;
    size_t start = 0;
    if(s[0] == '+') start = 1; // Optionanlly accept plus sign
    if(start == s.size()) return false; // But not only a plus sign
    for (size_t i = start ; i < s.size(); i++) {
        if(!isdigit(s[i])) return false; // Must be digits
    }
    return true;
}

// Validate double argument
bool is_double(const string& s) {
    try {
        size_t pos;
        stod(s, &pos);
        return pos == s.size();
    } catch(...) {
        return false;
    }
}

// Validate boolean argument
bool is_boolean(const string& s) {
    return (s == "true" || s == "false");
}

// Parse integer argument (if not given then use default parameter)
int get_int_arg(unordered_map<string, string>& args, string flag, int default_val) {
    // Search if flag is given from cmd
    auto it = args.find(flag);
    if(it == args.end()) {
        // If not then give default value
        return default_val;
    }

    // Check if it is the correct data type
    if(!is_int(it->second)) {
        throw runtime_error("Invalid input: argument " + flag + " requires non-negative integer value.");
    }

    // Return the value matched to the flag
    return stoi(it->second);
}

// Parse double argument (if not given then use default parameter)
double get_double_arg(unordered_map<string, string>& args, string flag, double default_val) {
    // Search if flag is given from cmd
    auto it = args.find(flag);
    if(it == args.end()) {
        // If not then give default value
        return default_val;
    }
    // Check if it is the correct data type
    if(!is_double(it->second)) {
        throw runtime_error("Invalid input: argument " + flag + " requires double value.");
    }

    // Return the value matched to the flag
    return stod(it->second);
}

// Parse string argument (if not given then use default parameter)
string get_string_arg(unordered_map<string, string>& args, string flag, string default_val) {
    // Search if flag is given from cmd
    auto it = args.find(flag);
    if(it == args.end()) {
        // If not then give default value
        return default_val;
    }

    if(it->second == "") {
        throw runtime_error("Invalid input: argument " + flag + " cannot be empty.");
    }

    // Return the value matched to the flag
    return it->second;
}

// Parse boolean argument (if not given then use default parameter)
bool get_boolean_arg(unordered_map<string, string>& args, string flag, bool default_val) {
    // Search if flag is given from cmd
    auto it = args.find(flag);
    if(it == args.end()) {
        // If not then give default value
        return default_val;
    }
    // Check if it is the correct data type
    if(!is_boolean(it->second)) {
        throw runtime_error("Invalid input: argument " + flag + " requires boolean value [true|false].");
    }

    // Convert from string to boolean
    if(it->second == "false") {
        return false;
    }
    return true;
}