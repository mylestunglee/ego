#include <limits>
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
	results_t& results_old,
	results_t& results_new,
	Evaluator& evaluator,
	config_t config) :
	evaluator(evaluator),
	config(config),
	max_trials(config.max_trials),
	sig_level(config.sig_level),
	boundaries(config.boundaries),
	is_discrete(config.is_discrete),
	fitness_percentile(config.fitness_percentile),
	results_old(results_old),
	results_new(results_new) {
	sort(results_old.begin(), results_old.end(), fitness_more_than);
	rng = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng, time(NULL));
}

Transferrer::~Transferrer() {
	gsl_rng_free(rng);
}

// Performs the automatic knowledge transfer
void Transferrer::run() {
	boundaries_t boundaries_old = infer_boundaries(results_old);

	// Prune EGO sampling for pre-sampled region
	boundaries_t presampled_space;
	if (!results_new.empty()) {
		presampled_space  = infer_boundaries(results_new);
	}

	auto samples = sample_results_old();
	assert(!samples.empty() || count_common_results({results_old, results_new}) >= 3);

	// Good for non-near-constrainted optimal solutions
	cout << "Performing multiquadratic regression fit" << endl;
	auto fs = multiquadratic_result_extrapolate(results_old);
	if (!fs.empty()) {
		// Guess optimial point
		auto mq_x = minimise_multiquadratic(fs, boundaries);
		mq_x = is_discrete ? round_vector(mq_x) : mq_x;
		auto mq_y = evaluator.evaluate(mq_x);
		results_new.push_back(make_pair(mq_x, mq_y));

		// Prune excess extrapolation
		auto spearmans = calc_spearmans(results_old);
		boundaries = prune_boundaries(boundaries, boundaries_old, fs,
			spearmans, sig_level);
	}

	cout << "Sampling new design space" << endl;
	// Compute fitness for sample
	for (auto result_old : samples) {
		auto x = result_old.first;
		auto y = evaluator.evaluate(x);

		assert(!y.empty());
		results_new.push_back(make_pair(x, y));

		animation_step();
	}

	// Define sampling rejection region
	boundaries_t rejection = get_intersection(boundaries_old, boundaries);
	if (rejection.empty()) {
		rejection = presampled_space;
	}

	EGO ego(evaluator, config, rejection, results_old);

	for (auto result_new : results_new) {
		ego.simulate(result_new.first, result_new.second);
	}

	cout << "Sampling using LHS" << endl;
	ego.sample_latin(5 * boundaries.size());
	cout << "Sampling using uniform" << endl;
	double bhv = calc_hypervolume(boundaries);
	double rhv = calc_hypervolume(rejection);
	double sample_ratio = (bhv - rhv) / bhv;
	ego.sample_uniform(5.0 * (double) boundaries.size() * sample_ratio);
	cout << "Using EGO" << endl;
	ego.run();
}

// Selects a subset of old_results to determine relationship of old and new evaluators
results_t Transferrer::sample_results_old() {
	vector<pair<vector<double>, vector<double>>> result;
	set<vector<double>> sampled;

	boundaries_t presampled_space;
	if (!results_new.empty()) {
		presampled_space = infer_boundaries(results_new);
	}

	// fitness_percentile is a hyperparameter for upper limit of samplable
	// fitnesses to minimise noise
	double fitness_threshold = calc_fitness_percentile(fitness_percentile);

	for (size_t trial = 0; trial < max_trials; trial++) {
		auto sample = results_old[gsl_rng_uniform_int(rng, results_old.size())];
		auto x = sample.first;
		auto y = sample.second;

		// Don't samples with bad fitnesses
		if (y[FITNESS_INDEX] > fitness_threshold) {
			continue;
		}

		// Only add sample when sample fits new parameter space and has not been
		// sampled before
		if (!is_bounded(x, boundaries) || sampled.find(x) != sampled.end()) {
			continue;
		}

		// Do not sample in pre-sampled space, assume we know the distribution
		if (!presampled_space.empty() && is_bounded(x, presampled_space)) {
			continue;
		}

		// Don't sample failed results
		if (y[LABEL_INDEX] != 0.0) {
			continue;
		}

		// Only sample iff is_discrete_old == is_discrete_new
		if (is_discrete) {
			auto rounded = round_vector(x);
			if (!equal(x.begin(), x.end(), rounded.begin())) {
				continue;
			}
		}

		sampled.insert(x);
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

	return (x.second)[FITNESS_INDEX] < (y.second)[FITNESS_INDEX];
}

// Given a value x between 0 and 1, determine the x percentile of the old
// fitness function. Requires results_old to be sorted
double Transferrer::calc_fitness_percentile(double percentile) {
	assert(0.0 <= percentile && percentile <= 1.0);

	// Find the number of valid samples
	size_t successes = 0;
	for (auto result_old : results_old) {
		if (result_old.second[LABEL_INDEX]) {
			successes++;
		}
	}

	assert (successes > 0);

	return results_old[percentile * (successes - 1.0)].second[FITNESS_INDEX];
}
