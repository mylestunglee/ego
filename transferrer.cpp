#include "transferrer.hpp"
#include "csv.hpp"
#include "surrogate.hpp"
#include "functions.hpp"
#include "ego.hpp"
#include "animation.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <assert.h>
#include <set>
#include <time.h>
#include <gsl_randist.h>

using namespace std;

const unsigned FITNESS_INDEX = 0;
const unsigned LABEL_INDEX   = 1;

Transferrer::Transferrer(
	string filename_results_old,
	Evaluator& evaluator,
	size_t max_evaluations,
	size_t max_trials,
	double convergence_threshold,
	double sig_level,
	boundaries_t boundaries,
	bool is_discrete,
	size_t constraints,
	size_t costs,
	double fitness_percentile) :
	evaluator(evaluator),
	max_evaluations(max_evaluations),
	max_trials(max_trials),
	convergence_threshold(convergence_threshold),
	sig_level(sig_level),
	boundaries(boundaries),
	is_discrete(is_discrete),
	constraints(constraints),
	costs(costs),
	fitness_percentile(fitness_percentile) {
	read_results(filename_results_old);
	sort(results_old.begin(), results_old.end(), fitness_more_than);
	rng = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng, time(NULL));
}

Transferrer::~Transferrer() {
	delete rng;
}

// Performs the automatic knowledge transfer
void Transferrer::run() {
	boundaries_t boundaries_old = infer_boundaries(results_old);

	results_t results_new;

	// Good for non-near-constrainted optimal solutions
	animation_start("Performing multiquadratic regression fit", 0, 1);
	auto fs = multiquadratic_result_extrapolate(results_old, constraints, costs);
	if (!fs.empty()) {
		// Guess optimial point
		auto mq_x = minimise_multiquadratic(fs, boundaries);
		auto mq_y = evaluator.evaluate(mq_x);
		results_new.push_back(make_pair(mq_x, mq_y));

		// Prune excess extrapolation
		auto spearmans = calc_spearmans(results_old);
		boundaries = prune_boundaries(boundaries, boundaries_old, fs,
			spearmans, sig_level);
	}
	animation_finish();

	auto samples = sample_results_old();
	assert(!samples.empty());

	animation_start("Sampling new design space:", 0, samples.size());
	set<pair<vector<double>, double>> old;

	// Compute fitness for sample
	for (auto result_old : samples) {
		auto x = result_old.first;
		auto y = evaluator.evaluate(x);

		assert(!y.empty());
		results_new.push_back(make_pair(x, y));
		old.insert(make_pair(x, y[0]));

		animation_step();
	}

	// Define sampling rejection region
	boundaries_t rejection;
	if (is_subset(boundaries, boundaries_old)) {
		rejection = get_intersection(boundaries_old, boundaries);
	}

	EGO ego(evaluator, boundaries, rejection, max_evaluations, max_trials,
		convergence_threshold, is_discrete, constraints, costs, old);

	for (auto result_new : results_new) {
		ego.simulate(result_new.first, result_new.second);
	}

	cout << "Sampling using LHS" << endl;
	ego.sample_latin(5 * boundaries.size());
	cout << "Sampling using uniform" << endl;
	ego.sample_uniform(5 * boundaries.size());
	cout << "Using EGO" << endl;
	ego.run();
}

// Reads results from a CSV file from a previous EGO computation
void Transferrer::read_results(string filename) {
	vector<vector<string>> data = read(filename);

	for (vector<string> line : data) {
		vector<double> x;
		vector<double> y;
		for (size_t i = 0; i < line.size(); i++) {
			double cell = stof(line[i]);
			if (i < line.size() - 2 - constraints - costs) {
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
	vector<pair<vector<double>, vector<double>>> result;
	set<vector<double>> sampled;
	// fitness_percentile is a hyperparameter for upper limit of samplable
	// fitnesses to minimise noise
	double fitness_threshold = calc_fitness_percentile(fitness_percentile);

	for (size_t trial = 0; trial < max_trials; trial++) {
		auto sample = results_old[gsl_rng_uniform_int(rng, results_old.size())];

		// auto sample = results_old[(size_t) gsl_ran_exponential(rng, 1.0) % results_old.size()];

		// Don't samples with bad fitnesses
		if (sample.second[FITNESS_INDEX] > fitness_threshold) {
			continue;
		}

		// Only add sample when sample fits new parameter space and has not been
		// sampled before
		if (!is_bounded(sample.first, boundaries) ||
			sampled.find(sample.first) != sampled.end()) {
			continue;
		}

		// Don't sample failed results
		if (!is_success(sample.second, constraints, costs)) {
			continue;
		}

		// Only sample iff is_discrete_old == is_discrete_new
		if (is_discrete) {
			auto x = sample.first;
			auto rounded = round_vector(x);
			if (!equal(x.begin(), x.end(), rounded.begin())) {
				continue;
			}
		}
		sampled.insert(sample.first);
		result.push_back(sample);
		// Sample at most 10 * dimension, but aim for 5 * dimension
		if (sampled.size() > 5 * boundaries.size()) {
			break;
		}
	}

	sort(result.begin(), result.end(), fitness_more_than);
	return result;
}

bool Transferrer::fitness_more_than(
	pair<vector<double>, vector<double>> x,
	pair<vector<double>, vector<double>> y
) {
	bool success_x = (x.second)[LABEL_INDEX] == 0.0;
	bool success_y = (y.second)[LABEL_INDEX] == 0.0;
	if (success_x && !success_y) {
		return true;
	}
	if (!success_x && success_y) {
		return false;
	}

	return (x.second)[0] < (y.second)[0];
}

// Calculates the average p-value from confidence interval tests on all samples
double Transferrer::calc_label_correlation(results_t results_new) {
	assert(!results_new.empty());

	GaussianProcess surrogate(boundaries.size());

	// Converts labels into continuous values
	for (auto result_old : results_old) {
		surrogate.add(
			result_old.first,
			(result_old.second)[LABEL_INDEX] == 0.0 ? 1.0 : 0.0);
	}

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

// Test each correlation score for best polynomial function
vector<double> Transferrer::test_correlation(vector<double> xs,
	vector<double> ys) {
	double pearson;
	double spearman;
	calc_correlation(xs, ys, pearson, spearman);

	if (isnan(pearson) || isnan(spearman)) {
		return {};
	}

	// If hypothesis test for linear relationship between ys_old and ys_new passes
	if (1.0 - pearson <= sig_level || 1.0 + pearson <= sig_level) {
		return fit_polynomial(xs, ys, 1);
	}

	// If hypothesis test for monotonic relationship passes
	if (1.0 - spearman <= sig_level || 1.0 + spearman <= sig_level) {
		return fit_polynomial(xs, ys, 2);
	}

	return {};
}

// Given a value x between 0 and 1, determine the x percentile of the old
// fitness function. Requires results_old to be sorted
double Transferrer::calc_fitness_percentile(double percentile) {
	assert(0.0 <= percentile && percentile <= 1.0);

	// Find the number of valid samples
	size_t successes = 0;
	for (auto result_old : results_old) {
		if (is_success(result_old.second, constraints, costs)) {
			successes++;
		}
	}

	assert(successes > 0);
	return results_old[percentile * (successes - 1.0)].second[FITNESS_INDEX];
}
