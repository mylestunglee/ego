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

	if (config_old.boundaries.size() != config_new.boundaries.size()) {
		return 0.0;
	}

	double boundaries_score = calc_boundaries_comparison_score(
		config_new.boundaries, config_old.boundaries);
	double cross_validation_score =
		calc_cross_validation_comparison_score(results_new, results_old);
	double results_score = calc_results_comparison_score(
		results_new, results_old);
	double names_score = calc_names_comparison_score(config_new.names,
		config_old.names);

/*	cout << "boundaries_score = " << boundaries_score << endl <<
			"cross_validation_score = " << cross_validation_score << endl <<
			"results_score = " << results_score << endl <<
			"names_score = " << names_score << endl;*/

	return names_score * (results_score + cross_validation_score + boundaries_score);
}

// Calculates a score representing the similiarity of two boundaries
double calc_boundaries_comparison_score(boundaries_t& boundaries_new,
	boundaries_t& boundaries_old) {
	boundaries_t intersection = get_intersection(boundaries_new,
		boundaries_old);
	double bhv = calc_hypervolume(boundaries_new);
	double ihv = calc_hypervolume(intersection);
	return ihv / bhv;
}

// Calculates a score representing the ability of a trasferred Gaussian process
double calc_cross_validation_comparison_score(results_t& results_new,
	results_t& results_old) {

	// Build prior samples
    set<pair<vector<double>, double>> added;
    for (auto result_old : results_old) {
        auto y = result_old.second;
        if (y[LABEL_INDEX] != 1.0) {
            added.insert(make_pair(result_old.first, y[FITNESS_INDEX]));
        }
    }

    TransferredGaussianProcess tgp(added);
    add_results_to_surrogate(results_new, tgp);

	// TODO: use tgp.cross_validate()?
	return 1.0 / tgp.cross_validate_parameter();
}

// Calculates a score based on the number of common designs
double calc_results_comparison_score(results_t& results_new, results_t& results_old) {
	size_t common = count_common_results({results_old, results_new});
	size_t bound = min(results_new.size(), results_old.size());
	return (double) common / (double) bound;
}

// Calcuates how similar two designs' parameters are
double calc_names_comparison_score(vector<string>& names_new,
	vector<string>& names_old) {
	if (names_new == names_old) {
		return 1.0;
	}
	return 0.0;
}
