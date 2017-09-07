#pragma once
#include <vector>
#include "functions.hpp"
#include "surrogate.hpp"

using namespace std;

void compare(config_t config_new, results_t& results_new,
	vector<config_t>& configs_old, vector<results_t>& results_olds,
	char* argv[]);

double calc_comparison_score(config_t& config_new, results_t& results_new,
	config_t& config_old, results_t& results_old);

double calc_boundaries_comparison_score(boundaries_t& boundaries_new,
	boundaries_t& boundaries_old);

double calc_cross_validation_comparison_score(results_t& results_new,
	results_t& results_old);

double calc_results_comparison_score(results_t& results_new,
	results_t& results_old);

double calc_names_comparison_score(vector<string>& names_new,
	vector<string>& names_old);

bool are_unique_names(vector<string> names);

vector<double> swap_pattern_point(vector<string> names_old,
	vector<string> names_new, vector<double> coeffss);

vector<double> calc_repository_midpoint(vector<config_t> config_old,
	config_t config_new, vector<results_t>& resultss);

double calc_tags_comparison_score(vector<string>& tags_new,
	vector<string>& tags_old);
