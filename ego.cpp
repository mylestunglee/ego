#include "ego.hpp"
#include "functions.hpp"
#include "animation.hpp"
#include "tgp.hpp"
#include <limits>
#include <thread>
#include <iostream>
#include <gsl_cdf.h>
#include <gsl_randist.h>
#include <time.h>
#include <functional>

using namespace std;

EGO::EGO(
	Evaluator& evaluator,
	boundaries_t boundaries,
	boundaries_t rejection,
	size_t max_evaluations,
	size_t max_trials,
	double convergence_threshold,
	bool is_discrete,
	size_t constraints,
	size_t costs,
	results_t results_old) :

	dimension(boundaries.size()),
	boundaries(boundaries),
	rejection(rejection),
	max_evaluations(max_evaluations),
	evaluations(0),
	max_trials(max_trials),
	convergence_threshold(convergence_threshold),
	is_discrete(is_discrete),
	x_opt({}),
	y_opt(numeric_limits<double>::max()),
	evaluator(evaluator) {

	// Initialise random numbers
	rng = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng, time(NULL));

	// If not using knowledge transfer
	if (results_old.empty()) {
		sg = new GaussianProcess(dimension);
		sg_label = new GaussianProcess(dimension);

		for (size_t constraint = 0; constraint < constraints; constraint++) {
			this->constraints.push_back(new GaussianProcess(dimension));
		}

		for (size_t cost = 0; cost < costs; cost++) {
			this->costs.push_back(new GaussianProcess(dimension));
		}
		return;
	}

	set<pair<vector<double>, double>> added_fitness;
	set<pair<vector<double>, double>> added_label;
	vector<set<pair<vector<double>, double>>> added_constraints =
		vector<set<pair<vector<double>, double>>>(constraints,
			set<pair<vector<double>, double>>{});
	vector<set<pair<vector<double>, double>>> added_costs =
		vector<set<pair<vector<double>, double>>>(costs,
			set<pair<vector<double>, double>>{});

		// Extract results for transfer
	for (auto result_old : results_old) {
		const size_t FITNESS_INDEX = 0;
		const size_t LABEL_INDEX = 1;
		const size_t FITNESS_LABEL_OFFSET = 2;
		auto x = result_old.first;
		auto y = result_old.second;

		added_label.insert(make_pair(x, y[LABEL_INDEX]));

		// Do not transfer non-label results if label failed
		if (y[LABEL_INDEX] == 1.0) {
			continue;
		}

		added_fitness.insert(make_pair(x, y[FITNESS_INDEX]));
		for (size_t i = 0; i < constraints; i++) {
			added_constraints[i].insert(
				make_pair(x, y[FITNESS_LABEL_OFFSET + i]));
		}

		for (size_t i = 0; i < costs; i++) {
			added_costs[i].insert(
				make_pair(x, y[FITNESS_LABEL_OFFSET + constraints + i]));
		}
	}

	// Build surrogates from previous data
	sg = new TransferredGaussianProcess(added_fitness);
	sg_label = new TransferredGaussianProcess(added_label);
	for (size_t i = 0; i < constraints; i++) {
		this->constraints.push_back(
			new TransferredGaussianProcess(added_constraints[i]));
	}

	for (size_t i = 0; i < costs; i++) {
		this->costs.push_back(
			new TransferredGaussianProcess(added_costs[i]));
	}
}

EGO::~EGO() {
	gsl_rng_free(rng);

	delete sg;
	delete sg_label;

	for (auto& constraint : constraints) {
		delete constraint;
	}

	for (auto& cost : costs) {
		delete cost;
	}
}

void EGO::run()
{
	if (x_opt.empty()) {
		return;
	}

	assert(evaluations > 0);

	sg->optimise();

	// Due to the discrete use of EGO, do not maximise EI if previously
	// evaluated
	function<bool (vector<double> x)> pred =
		[&](vector<double> x) {return !this->evaluator.was_evaluated(x);};

	while(evaluations < max_evaluations) {

		// Find a point with the highest expected improvement
		animation_start("Maximising expected improvement:", 0, max_trials);
		double neg_max_ei = numeric_limits<double>::max();
		vector<double> x = minimise(expected_improvement_bounded,
			generate_random_point, this, convergence_threshold, max_trials,
			pred, neg_max_ei);
		double max_ei = -neg_max_ei;

		if (x.empty()) {
			return;
		} else if (max_ei <= convergence_threshold) {
			cout << "Optimal found!" << endl;
			return;
		}

		// Ensure x is discrete if specified
		x = is_discrete ? round_vector(x) : x;

		// Evaluate new design and update GP models
		animation_start("Evaluating:", 0, 1);
		evaluate({x});
		animation_finish();
	}
}

// Evaluate samples from a Latin hypercube
void EGO::sample_latin(size_t n)
{
	auto xs = generate_latin_samples(rng, n, boundaries);
	reverse(xs.begin(), xs.end());
	vector<vector<double>> filered;
	for (auto x : xs) {
		x = is_discrete ? round_vector(x) : x;
		if (!is_bounded(x, rejection)) {
			filered.push_back(x);
		}
	}
	evaluate(filered);
}

// Takes random uniformly distributed samples
void EGO::sample_uniform(size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (size_t trial = 0; trial < max_trials; trial++) {
			auto x = generate_uniform_sample(rng, boundaries);
			x = is_discrete ? round_vector(x) : x;
			// Predicted label is valid and is not excluded
			if (success_probability(sg_label->mean(x), sg_label->sd(x)) >= 0.5 &&
				!is_bounded(x, rejection)) {
				evaluate({x});
				// Find next point
				break;
			}
		}
	}

	// log_surrogate_predictions(*sg, "predictions.csv", boundaries);
}

// Calcautes the expected improvement for maximisation with domain-specific knowledge
double EGO::expected_improvement_bounded(const gsl_vector* v, void* p) {
	EGO* ego = (EGO*) p;

	vector<double> x = gsl_to_std_vector(v);
	assert(x.size() == ego->dimension);

	x = ego->is_discrete ? round_vector(x) : x;

	// Negate to maximise expected_improvement because this function is called by
	// a minimiser
	auto expectation =
		-expected_improvement(ego->sg->mean(x), ego->sg->sd(x), ego->y_opt) *
		success_probability(ego->sg_label->mean(x), ego->sg_label->sd(x)) *
		ego->success_constraints_probability(x) / ego->predict_cost(x);

	if (isnan(expectation) || expectation == 0.0 ||
		!is_bounded(x, ego->boundaries)) {
		return euclidean_distance(x, ego->x_opt);
	}

	return expectation;
}

// Computes raw expected improvement
double EGO::expected_improvement(double y, double sd, double y_min) {
	if (sd <= 0.0) {
		return 0.0;
	}

	double y_diff = y_min - y;
	double y_diff_s = y_diff / sd;
	return y_diff * gsl_cdf_ugaussian_P(y_diff_s) + sd * gsl_ran_ugaussian_pdf(y_diff_s);
}

// Evaluates a vector to add to the training set
void EGO::thread_evaluate(EGO* ego, vector<double> x) {
	assert(ego != NULL);

	vector<double> y = ego->evaluator.evaluate(x);

	ego->evaluator_lock.lock();
	ego->simulate(x, y);
	ego->evaluator_lock.unlock();
}

// Evaluates multiple points xs
void EGO::evaluate(vector<vector<double>> xs) {
	for (auto x : xs) {
		thread_evaluate(this, x);
	}

	/* Concurrent method:
	vector<thread> threads;
	for (auto x : xs) {
		threads.push_back(thread(thread_evaluate, this, x));
	}

	for (auto& t : threads) {
		t.join();
	}*/
}

// Simulates an evaluation
void EGO::simulate(vector<double> x, vector<double> y) {
	const size_t FITNESS_INDEX = 0;
	const size_t LABEL_INDEX = 1;
	const size_t FITNESS_LABEL_OFFSET = 2;
	assert(y.size() == FITNESS_LABEL_OFFSET + constraints.size() + costs.size());

	evaluations++;

	evaluator.simulate(x, y);

	assert(y[LABEL_INDEX] == 0.0 || y[LABEL_INDEX] == 1.0 || y[LABEL_INDEX] == 2.0);

	// Update success GP
	sg_label->add(x, y[LABEL_INDEX] == 0.0 ? 1.0 : 0.0);

	// Do not update resource GPs if evaluation failed
	if (y[LABEL_INDEX] != 1.0) {
		sg->add(x, y[FITNESS_INDEX]);

		for (size_t constraint = 0; constraint < constraints.size(); constraint++) {
			double utilisation = y[FITNESS_LABEL_OFFSET + constraint];
			constraints[constraint]->add(x, utilisation);
		}

		for (size_t cost = 0; cost < costs.size(); cost++) {
			costs[cost]->add(x, y[FITNESS_LABEL_OFFSET + constraints.size() + cost]);
		}

		// Update optimal point and fitness if point is successful
		if (is_success(y, constraints.size(), costs.size()) &&
			y[FITNESS_INDEX] < y_opt) {
			x_opt = x;
			y_opt = y[0];
		}
	}

	cout << "Iteration [" << evaluations << "/" << max_evaluations << "]: ";
	print_vector(x);
	if (y_opt == numeric_limits<double>::max()) {
		cout << endl;
	} else {
		cout << "\t f";
		print_vector(x_opt);
		cout << " = " << y_opt << endl;
	}
}

// Given a point, predict the cost of evaluation
double EGO::predict_cost(vector<double> x) {
	double sum = 1.0;
	for (auto& cost : costs) {
		sum += max(cost->mean(x), 0.0);
	}

	assert(sum >= 1.0);
	return sum;
}

// Calculates the probabilty that a point is successful using prediction
// of constraints
double EGO::success_constraints_probability(vector<double> x) {
	double probability = 1.0;
	for (auto& constraint : constraints) {
		double mean = constraint->mean(x);
		double sd = constraint->sd(x);
		probability *= gsl_cdf_gaussian_Q(1.0 - mean, sd) / gsl_cdf_gaussian_Q(-mean, sd);
	}
	return probability;
}

// Generates a random point
vector<double> EGO::generate_random_point(void *p) {
	EGO* ego = (EGO*) p;
	return generate_uniform_sample(ego->rng, ego->boundaries);
}
