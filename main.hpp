#pragma once
#include <string>
#include "functions.hpp"

using namespace std;

int main(int argc, char* argv[]);
void print_help();
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

enum class Mode {
	undefined,
	optimise,
	transfer,
	compare
};
