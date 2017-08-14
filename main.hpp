#pragma once
#include <string>
#include "functions.hpp"
#include "ego.hpp"

using namespace std;

enum class Mode {
	undefined,
	optimise,
	transfer,
	compare
};

int main(int argc, char* argv[]);

void print_help(ostream& cstr);

bool read_config(
	string filename,
	size_t& max_evaluations,
	size_t& max_trials,
	size_t& constraints,
	size_t& costs,
	double& convergence_threshold,
	double& sig_level,
	double& fitness_percentile,
	bool& is_discrete,
	boundaries_t& boundaries
);

void simulate_results(EGO& ego, results_t results);
