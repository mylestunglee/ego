#pragma once
#include <vector>
#include "functions.hpp"
#include "surrogate.hpp"

using namespace std;

void compare(config_t config_new, results_t& results_new,
	vector<config_t>& configs_old, vector<results_t>& results_olds,
	char* argv[]);

double calc_comparison_score(config_t& config_new, config_t& config_old);
