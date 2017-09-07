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

	vector<results_t> valid_results;
	vector<config_t> valid_configs;

	for (size_t i = 0; i < results_olds.size(); i++) {
		cout << "Comparing design " << i << ": " << argv[4 + 2 * i] << endl;

		auto& results_old = results_olds[i];
		auto& config_old = configs_old[i];
		assert(!results_old.empty());

		double score = calc_comparison_score(config_new, results_new, config_old,
			results_old);
		if (score > best_score) {
			best_score = score;
			best_i = i;
		}

		cout << "\tComparison score: " << score << endl;

		if (score > 0.0) {
			valid_results.push_back(results_old);
			valid_configs.push_back(config_old);
		}
	}

	size_t common_source_points = count_common_results(results_olds);
	cout << "Common points across source designs: " << common_source_points << endl;
	size_t common_all_points = count_common_results(results_olds, results_new);
	cout << "Common points across all designs: " << common_all_points << endl;

	// Suggests weighted midpoint
//	print_vector(calc_repository_midpoint(valid_configs, config_new, valid_results));

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

	double names_score = calc_names_comparison_score(config_new.names,
		config_old.names);
	double boundaries_score = calc_boundaries_comparison_score(
		config_new.boundaries, config_old.boundaries);
	double cross_validation_score =
		calc_cross_validation_comparison_score(results_new, results_old);
	double results_score = calc_results_comparison_score(
		results_new, results_old);
	double tags_score = calc_tags_comparison_score(config_new.tags,
		config_old.tags);

	return names_score * (results_score + cross_validation_score +
		boundaries_score + tags_score);
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

// Returns true iff each name in names is unique
bool are_unique_names(vector<string> names) {
	set<string> seen;
	for (auto name : names) {
		if (seen.find(name) != seen.end()) {
			return false;
		}
		seen.insert(name);
	}
	return true;
}

// Applys the swap pattern from names_old to names_new with coeffss
vector<double> swap_pattern_point(vector<string> names_old,
	vector<string> names_new, vector<double> x) {
	vector<double> result;
	size_t n = x.size();

	assert(n == names_old.size());
	assert(n == names_new.size());
	assert(are_unique_names(names_old));
	assert(are_unique_names(names_new));

	for (size_t i = 0; i < n; i++) {
		size_t j = 0;
		// Find matching name in old names
		for (; j < n; j++) {
			if (names_new[i] == names_old[j]) {
				break;
			}
		}
		result.push_back(x[j]);
	}
	return result;
}

// Compute the weighted midpoint of a multi-quadratic regression
vector<double> calc_repository_midpoint(vector<config_t> config_old,
	config_t config_new, vector<results_t>& resultss) {

	size_t n = config_old.size();
	assert(n == resultss.size());

	vector<vector<double>> minimas;
	vector<vector<double>> spearmanss;

	// Collect statistics
	for (size_t i = 0; i < n; i++) {
		auto& results = resultss[i];
		auto coeffss = multiquadratic_result_extrapolate(results);
		auto unswapped = minimise_multiquadratic(coeffss,
			config_new.boundaries);
		auto minima = swap_pattern_point(config_new.names, config_old[i].names,
			unswapped);
		auto spearmans = calc_spearmans(results);
		minimas.push_back(minima);
		spearmanss.push_back(spearmans);
	}

	size_t dimension = config_new.boundaries.size();
	vector<double> weighted_minima;

	for (size_t i = 0; i < dimension; i++) {
		double sum_minima = 0.0;
		double sum_normaliser = 0.0;
		for (size_t j = 0; j < resultss.size(); j++) {
			sum_minima += minimas[j][i] * abs(spearmanss[j][i]);
			sum_normaliser += abs(spearmanss[j][i]);
		}
		weighted_minima.push_back(sum_minima / sum_normaliser);
	}

	return weighted_minima;
}

// Score of number of common tags
double calc_tags_comparison_score(vector<string>& tags_new,
	vector<string>& tags_old) {
	set<string> seen;
	for (auto& tag_new : tags_new) {
		seen.insert(tag_new);
	}
	size_t common = 0;
	for (auto& tag_old : tags_old) {
		if (seen.find(tag_old) != seen.end()) {
			common += 1;
		}
	}
	size_t max_common = min(tags_new.size(), tags_old.size());
	if (common == 0 || max_common == 0) {
		return 0.0;
	}
	return (double) common / (double) max_common;
}
