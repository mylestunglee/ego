#include <iostream>
#include <set>
#include <limits>
#include "compare.hpp"
#include "tgp.hpp"
#include "csv.hpp"

using namespace std;

const size_t FITNESS_INDEX = 0;
const size_t LABEL_INDEX = 1;

// Attempts to find the best combination of results for knowledge transfer
void compare(config_t config_new, results_t& results_new,
	vector<config_t>& configs_old, vector<results_t>& results_olds,
	char* argv[]) {
	assert(!results_new.empty());

	double best_score = 0;
	size_t best_i = 0;

	// Statistics
	size_t dimension = config_new.boundaries.size();

	cout << "Base case" << endl;
	GaussianProcess gp(dimension);
	add_results_to_surrogate(results_new, gp);

	cout << "\tFitness cross-validation mean error: " << gp.cross_validate() << endl;

	for (size_t i = 0; i < results_olds.size(); i++) {
		cout << "Comparing design " << i << ": " << argv[4 + 2 * i] << endl;

		results_t& results_old = results_olds[i];
		assert(!results_old.empty());

		double score = calc_comparison_score(config_new, results_new, configs_old[i],
			results_old);
		if (score > best_score) {
			best_score = score;
			best_i = i;
		}

		cout << "\tComparison score: " << score << endl;
	}

	size_t common_source_points = count_common_results(results_olds);
	cout << "Common points across source designs: " << common_source_points << endl;
	size_t common_all_points = count_common_results(results_olds, results_new);
	cout << "Common points across all designs: " << common_all_points << endl;

	if (best_score > 0.0) {
		write("best_source.txt", {{to_string(best_i)}});
	}
}

// Compares how suitable knowledge transfer will be between the source and
// target configs
double calc_comparison_score(config_t& config_new, results_t& results_new,
	config_t& config_old, results_t& results_old) {

	size_t common_results = count_common_results({results_old, results_new});
	cout << "\tNumber of common points: " << common_results << endl;

	// Incompatible if different dimensions
	if (config_new.boundaries.size() != config_old.boundaries.size()) {
		return 0.0;
	}

	// Build tgp
	set<pair<vector<double>, double>> added;
	for (auto result_old : results_old) {
		auto y = result_old.second;
		if (y[LABEL_INDEX] != 1.0) {
			added.insert(make_pair(result_old.first, y[FITNESS_INDEX]));
		}
	}
	TransferredGaussianProcess tgp(added);
	add_results_to_surrogate(results_new, tgp);

	cout << "\tFitness cross-validation mean error: " <<
		tgp.cross_validate() << endl;
	double parameter_error = tgp.cross_validate_parameter();
	cout << "\tParameter cross-validation mean error: " << parameter_error
		<< endl;

	if (config_new.names == config_old.names) {
		return 1.0 + 1.0 / parameter_error;
	}

	// TODO: implement score formula
	return 0.0;
}
