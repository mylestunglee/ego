#include <limits>
#include "ego.hpp"
#include "functions.hpp"
#include <thread>
#include <iostream>
#include <iomanip>
#include <gsl_cdf.h>
#include <gsl_randist.h>
#include <gsl_multimin.h>

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
	size_t costs
) :	max_evaluations(max_evaluations),
	max_trials(max_trials),
	convergence_threshold(convergence_threshold),
	is_discrete(is_discrete),
	evaluator(evaluator)
{
	dimension = boundaries.size();
	this->boundaries = boundaries;
	this->rejection = rejection;

	sg = new Surrogate(dimension);
	sg_label = new Surrogate(dimension);

	for (size_t constraint = 0; constraint < constraints; constraint++) {
		this->constraints.push_back(new Surrogate(dimension));
	}

	for (size_t cost = 0; cost < costs; cost++) {
		this->costs.push_back(new Surrogate(dimension));
	}

	rng = gsl_rng_alloc(gsl_rng_taus);

	evaluations = 0;
	y_opt = numeric_limits<double>::max();
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
	assert(evaluations > 0);

	while(evaluations < max_evaluations) {
		sg->train();
		sg_label->train();

		// Find a point with the highest expected improvement
		double improvement = 0.0;
		vector<double> x = maximise_expected_improvement_global(improvement);

		if (x.empty()) {
			cout << "Cannot maximise expected improvement!" << endl;
			return;
		} else if (improvement <= convergence_threshold) {
			cout << "Optimial found!" << endl;
			return;
		}

		evaluate({x});
	}
}

// Evaluate samples from a Latin hypercube
void EGO::sample_latin(size_t n)
{
	auto xs = generate_latin_samples(rng, n, boundaries);
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
}

// Runs multiple trials to maximise global expected improvement
vector<double> EGO::maximise_expected_improvement_global(double& improvement) {
	double improvement_best = 0.0;
	vector<double> x_best;
	for (size_t trial = 0; trial < max_trials; trial++) {
		double improvement_trial = 0.0;
		auto x = maximise_expected_improvement_local(improvement_trial);
		if (improvement_trial > improvement_best) {
			improvement_best = improvement_trial;
			x_best = x;
		}
	}
	improvement = improvement_best;
	return x_best;
}

// Use the Nelder-Mead method to maximise local expected improvement
vector<double> EGO::maximise_expected_improvement_local(double &improvement) {
	const gsl_multimin_fminimizer_type *algorithm = gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_function func;

	// Starting point
	gsl_vector* x = gsl_vector_alloc (dimension);
	auto v = generate_uniform_sample(rng, boundaries);
	for (size_t i = 0; i < dimension; i++) {
		gsl_vector_set(x, i, v[i]);
	}

	// Set initial step sizes to 1
	gsl_vector* ss = gsl_vector_alloc (dimension);
	gsl_vector_set_all (ss, 1.0);

	// Initialize method and iterate
	func.n = dimension;
	func.f = &EGO::expected_improvement_bounded;
	func.params = this;

	gsl_multimin_fminimizer* s = gsl_multimin_fminimizer_alloc (algorithm, dimension);
	gsl_multimin_fminimizer_set (s, &func, x, ss);

	for (size_t trial = 0; trial < max_trials; trial++) {
		int status = gsl_multimin_fminimizer_iterate(s);

		if (status) {
			break;
		}

    	double size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, convergence_threshold);

		if (status != GSL_CONTINUE) {
			break;
		}
	}

	vector<double> result;

	for (size_t i = 0; i < dimension; i++) {
		result.push_back(gsl_vector_get(s->x, i));
	}
	result = is_discrete ? round_vector(result) : result;

	improvement = -s->fval;

	gsl_vector_free(x);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free (s);

	return result;
}

double EGO::expected_improvement_bounded(const gsl_vector* v, void* p) {
	EGO* ego = (EGO*) p;

	vector<double> x;
	for (size_t i = 0; i < ego->dimension; i++) {
		x.push_back(gsl_vector_get(v, i));
	}
	assert(x.size() == ego->dimension);

	x = ego->is_discrete ? round_vector(x) : x;

	// Negate to maximise expected_improvement because this function is called by
	// a minimiser
	auto expectation =
		-expected_improvement(ego->sg->mean(x), ego->sg->sd(x), ego->y_opt) *
		success_probability(ego->sg_label->mean(x), ego->sg_label->sd(x)) *
		ego->success_constraints_probability(x) / ego->predict_cost(x);

	if (expectation == 0.0 || !is_bounded(x, ego->boundaries)
		|| is_bounded(x, ego->rejection)) {
		return euclidean_distance(x, ego->x_opt);
	}

	return expectation;
}

double EGO::expected_improvement(double y, double sd, double y_min) {
	if (sd <= 0.0) {
		return 0.0;
	}

	double y_diff = y_min - y;
	double y_diff_s = y_diff / sd;
	return y_diff * gsl_cdf_ugaussian_P(y_diff_s) + sd * gsl_ran_ugaussian_pdf(y_diff_s);
}

/* Evaluates a vector to add to the training set */
void EGO::thread_evaluate(EGO* ego, vector<double> x) {
	assert(ego != NULL);

	vector<double> y = ego->evaluator.evaluate(x);

	ego->evaluator_lock.lock();
	ego->simulate(x, y);

	cout << "Iteration [" << ego->evaluations << "/" << ego->max_evaluations << "]: " << fixed;
	print_vector(x);
	cout << "\t f";
	print_vector(ego->x_opt);
	cout << " = " << setprecision(6) << ego->y_opt << endl;

	ego->evaluator_lock.unlock();
}

// Concurrently evaluates multiple points xs
void EGO::evaluate(vector<vector<double>> xs) {
	vector<thread> threads;

	for (auto x : xs) {
		threads.push_back(thread(thread_evaluate, this, x));
	}

	for (auto& t : threads) {
		t.join();
	}
}

// Simulates an evaluation
void EGO::simulate(vector<double> x, vector<double> y) {
	const size_t FITNESS_INDEX = 0;
	const size_t LABEL_INDEX = 1;
	const size_t FITNESS_LABEL_OFFSET = 2;
	assert(y.size() == FITNESS_LABEL_OFFSET + constraints.size() + costs.size());

	evaluations++;

	evaluator.simulate(x, y);

	// Update GPs
	sg->add(x, y[FITNESS_INDEX]);
	sg_label->add(x, y[LABEL_INDEX] == 0 ? 1.0 : 0.0);

	bool successful_constraint = true;

	for (size_t constraint = 0; constraint < constraints.size(); constraint++) {
		double utilisation = y[FITNESS_LABEL_OFFSET + constraint];
		constraints[constraint]->add(x, utilisation);
		if (utilisation > 1.0) {
			successful_constraint = false;
		}
	}

	for (size_t cost = 0; cost < costs.size(); cost++) {
		costs[cost]->add(x, y[FITNESS_LABEL_OFFSET + constraints.size() + cost]);
	}

	// Update optimal point and fitness if point is successful
	if (y[LABEL_INDEX] == 0 && successful_constraint && y[FITNESS_INDEX] < y_opt) {
		x_opt = x;
		y_opt = y[0];
	}
}

// Given a point, predict the cost of evaluation
double EGO::predict_cost(vector<double> x) {
	double sum = 1.0;
	for (auto& cost : costs) {
		sum += cost->mean(x);
	}
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
