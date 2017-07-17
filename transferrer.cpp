#include "transferrer.hpp"
#include "csv.hpp"
#include "surrogate.hpp"
#include "functions.hpp"
#include "ego.hpp"
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
	size_t costs) :
	evaluator(evaluator),
	max_evaluations(max_evaluations),
	max_trials(max_trials),
	convergence_threshold(convergence_threshold),
	sig_level(sig_level),
	boundaries(boundaries),
	is_discrete(is_discrete),
	constraints(constraints),
	costs(costs) {
	read_results(filename_results_old);
	cout << "Sorting old results" << endl;
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

	if (boundaries_old.size() < boundaries.size()) {
		extrude(boundaries_old);
		return;
	} else if (boundaries_old.size() > boundaries.size()) {
		reduce(boundaries_old);
		return;
	}

	vector<double> ys_old;
	vector<double> ys_new;
	results_t results_new;

	auto sample = results_old; // sample_results_old();
	assert(!sample.empty());

	cout << "Sampling new design space";

	// Compute fitness for sample
	for (auto result_old : sample) {
		auto x_old = result_old.first;
		auto y_old = result_old.second;
		auto y_new = evaluator.evaluate(x_old);

		if (y_old[LABEL_INDEX] != 1.0 && is_success(y_new, constraints, costs)) {
			ys_old.push_back(y_old[FITNESS_INDEX]);
			ys_new.push_back(y_new[FITNESS_INDEX]);
		}

		results_new.push_back(make_pair(x_old, y_new));
		cout << ".";
	}
	cout << endl;

	double label_correlation = calc_label_correlation(results_new);

	if (1.0 - label_correlation > sig_level) {
		cout << "Insufficent consistency of labels." << endl;
		return;
	}

	vector<double> coeffs = test_correlation(ys_old, ys_new);

	if (coeffs.empty()) {
		cout << "Insufficent monotonic relationship between fitness functions." << endl;
		return;
	}

	if (is_subset(boundaries, boundaries_old)) {
		interpolate({}, {0.0, 1.0}, results_new);
		return;
	}

	interpolate(boundaries_old, {0, 1.0}, results_new);
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
	// 0.15 is a hyperparameter for upper limit of samplable fitnesses to
	// minimise noise
	double fitness_threshold = calc_fitness_percentile(0.15);

	for (size_t trial = 0; trial < max_trials; trial++) {
		auto sample = results_old[gsl_rng_uniform_int(rng, results_old.size())];

		//auto sample = results_old[(size_t) gsl_ran_exponential(rng, 10) % results_old.size()];

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
		// Sample at most 10 * dimension
		if (sampled.size() > 10 * boundaries.size()) {
			return result;
		}
	}

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
	boundaries_t intersection;
	if (!boundaries_old.empty()) {
		boundaries_t intersection = get_intersection(boundaries_old, boundaries);
	}
	cout << "Simulating results" << endl;
	EGO ego(evaluator, boundaries, intersection, max_evaluations, max_trials,
		convergence_threshold, is_discrete, constraints, costs);
	for (auto result_new : results_new) {
		// Update old fitness to new fitness
		result_new.second[FITNESS_INDEX] = apply_polynomial(
			result_new.second[FITNESS_INDEX], coeffs);
		ego.simulate(result_new.first, result_new.second);
	}
	cout << "Sampling using LHS" << endl;
	ego.sample_latin(5 * boundaries.size());
	cout << "Sampling using uniform" << endl;
	ego.sample_uniform(5 * boundaries.size());
	ego.run();
}

// Given an old parameter space, find a higher-dimensional mapping from the old
// parameter space to the new parameter space
void Transferrer::extrude(boundaries_t boundaries_old) {
	size_t dimension_old = boundaries_old.size();
	size_t dimension = boundaries.size();
	boundaries_t space_common = boundaries_t(boundaries.begin(),
		boundaries.begin() + dimension_old);
	space_extend = boundaries_t(boundaries.begin() + dimension_old,
		boundaries.end());
	space_intersection = get_intersection(space_common, boundaries_old);
	boundaries_t space_extrude = join_boundaries(space_intersection, space_extend);

	// Test number of knowledge transferable points of the old boundaries
	size_t samples_old_count = 0;
	for (auto result_old : results_old) {
		if (is_bounded(result_old.first, space_intersection)) {
			samples_old_count++;
		}
	}
	if (samples_old_count < 3) {
		cout << "Too few common points to perform correlation analysis" << endl;
		return;
	}

	// Generate a model of the new design space
	auto samples_new = generate_latin_samples(rng, 5 * dimension, space_extrude);
	results_t results_new;
	predictor = new Surrogate(dimension);
	for (auto sample_new : samples_new) {
		auto y = evaluator.evaluate(sample_new);
		predictor->add(sample_new, y[FITNESS_INDEX]);
	}
	predictor->train();

	// Find the best cross section of some variable point in space_extend
	double neg_max_correlation = numeric_limits<double>::max();
	auto point = minimise(cross_section_correlation, generate_random_point,
		this, convergence_threshold, max_trials, neg_max_correlation);

	if (point.empty()) {
		cout << "Cannot maximise correlation across cross sections" << endl;
		return;
	}

	vector<double> fitnesses_old;
	vector<double> fitnesses_new;

	// Collect correlation data
	for (auto result_old : results_old) {
		auto x = result_old.first;
		auto y = result_old.second;
		if (is_bounded(x, space_intersection)) {
			fitnesses_old.push_back(y[FITNESS_INDEX]);
			fitnesses_new.push_back(
				predictor->mean(join_vectors(x, point)));
		}
	}

	// Bulid mapping function from old to new design
	auto coeffs = test_correlation(fitnesses_old, fitnesses_new);
	if (coeffs.empty()) {
		cout << "Insufficent correlation for any cross section" << endl;
		return;
	}

	// Add old samples as new correlated samples
	for (auto result_old : results_old) {
		auto x = result_old.first;
		auto y = result_old.second;
		y[FITNESS_INDEX] = apply_polynomial(y[FITNESS_INDEX], coeffs);
		if (is_bounded(x, space_intersection)) {
			results_new.push_back(make_pair(join_vectors(x, point), y));
		}
	}

	// Sample space outside of space_extend but inside boundaries
	EGO ego(evaluator, boundaries, {}, max_evaluations, max_trials,
		convergence_threshold, is_discrete, constraints, costs);
	for (auto result_new : results_new) {
		// Update old fitness to new fitness
		ego.simulate(result_new.first, result_new.second);
	}
	ego.sample_uniform(5 * boundaries.size());
	ego.run();

	delete predictor;
}

// Given an old parameter space, find a lower-dimensional mapping from the old
// parameter space to the new parameter space
void Transferrer::reduce(boundaries_t boundaries_old) {
	Surrogate surrogate(boundaries_old.size());

	// Rebuild previous GP
	for (auto result_old : results_old) {
		auto x = result_old.first;
		auto y = result_old.second;
		surrogate.add(x, y[FITNESS_INDEX]);
	}
	surrogate.train();
}

// Generates a random point in space_new
vector<double> Transferrer::generate_random_point(void* p) {
	Transferrer* t = (Transferrer*) p;
	return generate_uniform_sample(t->rng, t->space_extend);
}

double Transferrer::cross_section_correlation(const gsl_vector* v, void* p) {
	Transferrer* t = (Transferrer*) p;
	auto extend = gsl_to_std_vector(v);

	if (!is_bounded(extend, t->space_extend)) {
		return euclidean_distance(extend, vector<double>(extend.size(), 0.0));
	}

	vector<double> fitnesses_old;
	vector<double> fitnesses_new;

	// Collect correlation data
	for (auto result_old : t->results_old) {
		auto x = result_old.first;
		auto y = result_old.second;
		if (is_bounded(x, t->space_intersection)) {
			fitnesses_old.push_back(y[FITNESS_INDEX]);
			fitnesses_new.push_back(
				t->predictor->mean(join_vectors(x, extend)));
		}
	}

	// Perform correlation analysis
	double pearson;
	double spearman;
	calc_correlation(fitnesses_old, fitnesses_new, pearson, spearman);

	// If either old or new fitnesses have no variance then we have perfect
	// correlation
	if (isnan(pearson) || isnan(spearman)) {
		return -1.0;
	}

	// We try to find the best correlation using a minimiser
	return -max(abs(pearson), abs(spearman));
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
		return fit_polynomial_robust(xs, ys, 1);
	}

	// If hypothesis test for monotonic relationship passes
	if (1.0 - spearman <= sig_level || 1.0 + spearman <= sig_level) {
		return fit_polynomial_robust(xs, ys, 2);
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
	return results_old[percentile * (double) successes].second[FITNESS_INDEX];
}
