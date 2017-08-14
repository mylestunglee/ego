#pragma once
#include <vector>
#include "functions.hpp"
#include "surrogate.hpp"

using namespace std;

void compare(results_t& results_new, vector<results_t>& results_olds);

void add_results_to_surrogate(results_t& results, Surrogate& surrogate);
