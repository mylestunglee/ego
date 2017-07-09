#include "transferrer.hpp"
#include "csv.hpp"
#include "constants.hpp"
#include "surrogate.hpp"
#include "functions.hpp"
#include "ego.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <assert.h>
#include <set>

using namespace std;

Transferrer::Transferrer(
	string filename_results_old,
	Evaluator& evaluator,
	size_t max_evaluations,
	size_t max_trials,
	double convergence_threshold,
	double sig_level,
	boundaries_t boundaries
) : evaluator(evaluator),
	max_evaluations(max_evaluations),
	max_trials(max_trials),
	convergence_threshold(convergence_threshold),
	sig_level(sig_level),
	boundaries(boundaries) {
	read_results(filename_results_old);
	sort(results_old.begin(), results_old.end(), fitness_more_than);
}

// Performs the automatic knowledge transfer
void Transferrer::run() {
	vector<double> ys_old;
	vector<double> ys_new;
	results_t results_new;

	auto sample = sample_results_old();
	assert(!sample.empty());

	// Compute fitness for sample
	for (auto result_old : sample) {
		auto x_old = result_old.first;
		auto y_old = result_old.second;
		auto y_new = evaluator.evaluate(x_old);

		ys_old.push_back(y_old[FITNESS_INDEX]);
		ys_new.push_back(y_new[FITNESS_INDEX]);
		results_new.push_back(make_pair(x_old, y_new));
	}

	double label_correlation = calc_label_correlation(results_new);
	cout << "Label correlation: " << label_correlation << endl;

	if (1.0 - label_correlation > sig_level) {
		cout << "Insufficent consistency of labels." << endl;
		return;
	}

	double pearson;
	double spearman;
	calc_correlation(ys_old, ys_new, pearson, spearman);

	cout << "Pearson: " << pearson << ", Spearman: " << spearman << endl;
	double y_old_best = (results_old[0].second)[FITNESS_INDEX];
	vector<double> coeffs;

	// If hypothesis test for linear relationship between ys_old and ys_new passes
	if (1.0 - pearson <= sig_level || 1.0 + pearson <= sig_level) {
		coeffs = fit_polynomial(ys_old, ys_new, 1);
		cout << "Using linear regression" << endl;

	// If hypothesis test for monotonic relationship passes
	} else if (1.0 - spearman <= sig_level || 1.0 + spearman <= sig_level) {
		coeffs = fit_polynomial(ys_old, ys_new, 2);
		cout << "Using quadratic regression" << endl;
	} else {
		cout << "Insufficent monotonic relationship between fitness functions." << endl;
		return;
	}

	double y_new_best_approx = apply_polynomial(y_old_best, coeffs);
	boundaries_t boundaries_old = infer_boundaries(results_old);

	if (is_subset(boundaries, boundaries_old)) {
		cout << "Trivial mapping, y_new_opt = " << y_new_best_approx << endl;
		return;
	}

	cout << "Using interpolation: ";
	print_vector(coeffs);
	cout << endl;

	interpolate(boundaries_old, coeffs, results_new);
}

// Reads results from a CSV file from a previous EGO computation
void Transferrer::read_results(string filename) {
	vector<vector<string>> data = read(filename);

	for (vector<string> line : data) {
		vector<double> x;
		vector<double> y;
		for (size_t i = 0; i < line.size(); i++) {
			double cell = stof(line[i]);
			if (i < line.size() - 3) {
				x.push_back(cell);
			} else {
				y.push_back(cell);
			}
		}
		results_old.push_back(make_pair(x, y));
	}
}

// Selects a subset of old_results to determine relationship of old and new evaluators
results_t Transferrer::sample_results_old() {
	return results_old;

	vector<pair<vector<double>, vector<double>>> result;
	set<vector<double>> sampled;

	for (unsigned i = 0; i < SAMPLE_TRIALS; i++) {
		auto sample = results_old[rand() % results_old.size()];

		// Only add sample when sample fits new parameter space and has not been
		// sampled before
		if (is_bounded(sample.first, boundaries) &&
			sampled.find(sample.first) == sampled.end()) {
			sampled.insert(sample.first);
			result.push_back(sample);
		}

		if (sampled.size() > SAMPLE_MAX) {
			break;
		}
	}

	return result;
}

bool Transferrer::fitness_more_than(
	pair<vector<double>, vector<double>> x,
	pair<vector<double>, vector<double>> y
) {
	return (x.second)[0] < (y.second)[0];
}

// Calculates the average p-value from confidence interval tests on all samples
double Transferrer::calc_label_correlation(results_t results_new) {
	assert(!results_new.empty());

	Surrogate surrogate(boundaries.size());

	// Converts labels into continuous values
	for (auto result_old : results_old) {
		surrogate.add(
			result_old.first,
			(result_old.second)[LABEL_INDEX] == 0.0 ? 1.0 : 0.0);
	}
	surrogate.train();

	vector<double> coeffs;

	for (auto result_new : results_new) {
		auto x = result_new.first;
		double mean = surrogate.mean(x);
		double sd = surrogate.sd(x);

		// If GP prediction is certain, otherwise assume normal distribution
		if (isnan(sd) || sd <= 0.0) {
			coeffs.push_back(1.0);
		} else if (result_new.second[LABEL_INDEX] == 0.0) {
			coeffs.push_back(success_probability(mean, sd));
		} else {
			coeffs.push_back(1.0 - success_probability(mean, sd));
		}
	}

	return accumulate(coeffs.begin(), coeffs.end(), 0.0) / coeffs.size();
}

// Given a old parameter space, the approximation of y_olds to y_news, and some
// samples of the new space, find y_opt
void Transferrer::interpolate(boundaries_t boundaries_old, vector<double> coeffs,
	results_t results_new) {

	boundaries_t intersection = get_intersection(boundaries_old, boundaries);

	EGO ego(evaluator, boundaries, intersection, max_evaluations, max_trials,
		convergence_threshold);
	for (auto result_new : results_new) {
		// Update old fitness to new fitness
		result_new.second[FITNESS_INDEX] = apply_polynomial(
			result_new.second[FITNESS_INDEX], coeffs);
		ego.simulate(result_new.first, result_new.second);
	}
	ego.sample_uniform(10);
	ego.run();
}
