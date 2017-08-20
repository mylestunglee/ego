#include <iostream>
#include <set>
#include <limits>
#include "compare.hpp"
#include "tgp.hpp"

using namespace std;

const size_t FITNESS_INDEX = 0;
const size_t LABEL_INDEX = 1;

// Attempts to find the best combination of results for knowledge transfer
void compare(results_t& results_new, vector<results_t>& results_olds) {
	size_t dimension = results_new[0].first.size();

	cout << "Base case" << endl;
	GaussianProcess gp(dimension);
	add_results_to_surrogate(results_new, gp);

	cout << "\tFitness cross-validation mean error: " << gp.cross_validate() << endl;

	for (results_t& results_old : results_olds) {
		cout << "Comparing!" << endl;
		cout << "\tNumber of common points: " <<
			count_common_results({results_old, results_new}) << endl;

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
		cout << "\tParameter cross-validation mean error: " <<
			tgp.cross_validate_parameter() << endl;
	}

	size_t common_source_points = count_common_results(results_olds);
	cout << "Common points across source designs: " << common_source_points << endl;
	size_t common_all_points = count_common_results(results_olds, results_new);
	cout << "Common points across all designs: " << common_all_points << endl;

}
