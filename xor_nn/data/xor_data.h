#pragma once
#include <vector>

struct Sample {
    std::vector<double> input;   
    std::vector<double> output;  
};


std::vector<Sample> get_xor_data();
