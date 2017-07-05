#include <limits>
#include "ego.hpp"
#include "ihs.hpp"
#include "functions.hpp"
#include <thread>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_multimin.h>

using namespace std;

EGO::EGO(vector<pair<double, double>> boundaries, Evaluator& evaluator) :
	evaluator(evaluator)
{
	dimension = boundaries.size();
	this->boundaries = boundaries;

	sg = new Surrogate(boundaries.size(), SEiso, true, false);
	sg_cost = new Surrogate(boundaries.size(), SEard);

	rng = gsl_rng_alloc(gsl_rng_taus);

  max_evaluations = 1000;
  evaluations = 0;
	max_trials = 100;
	convergence_threshold = 0.01;
  best_fitness = std::numeric_limits<double>::max();
}

EGO::~EGO() {
	gsl_rng_free(rng);

	delete sg;
	delete sg_cost;
}

void EGO::run()
{
	assert(evaluations > 0);

	sg->train_gp_first();
	while(evaluations < max_evaluations) {
		sg->train();
		sg_cost->train();

		cout << "Iteration [" << (evaluations + 1) << "/" << max_evaluations << "] ";

		// Find a point with the highest expected improvement
		double improvement = 0.0;
		vector<double> x;
		for (size_t trial = 0; trial < max_trials; trial++) {
			x = maximise_expected_improvement(improvement);
			if (!x.empty()) {
				break;
			}
		}

		if (x.empty()) {
			cout << "Cannot maximise expected improvement!" << endl;
			return;
		} else if (improvement < convergence_threshold) {
			cout << "Optimial found!" << endl;
			return;
		}

		cout << "Evaluating: ";
		print_vector(x);

		evaluate({x});

		cout << "\tBest fitness: " << best_fitness << " at ";
		print_vector(best_particle);
		cout << endl;
	}
}

void EGO::sample_latin(size_t n)
{
	int seed = gsl_rng_get(rng);
	int* latin = ihs(dimension, n, 5, seed);
	assert(latin != NULL);

	vector<vector<double>> xs;

	// Scale latin hypercube to fit parameter space
	for (size_t i = 0; i < n; i++) {
		vector<double> x;
		for (size_t j = 0; j < (unsigned) dimension; j++) {
			double lower = boundaries[j].first;
			double upper = boundaries[j].second;
			double x_j = lower + (latin[i * dimension + j] - 1.0) / (n - 1.0) * (upper - lower);
			x.push_back(x_j);
		}
		xs.push_back(x);
	}

	delete latin;

	evaluate(xs);

	sg->choose_svm_param(5, true);
}

void EGO::sample_uniform(size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (size_t trial = 0; trial < max_trials; trial++) {
			auto x = generate_uniform_sample(rng, boundaries);

			// Predicted label is valid
			if (sg->svm_label(&x[0]) == 1) {
				evaluate({x});
				sg->choose_svm_param(5);

				// Find next point
				break;
			}
		}
	}
}

vector<double> EGO::maximise_expected_improvement(double &improvement) {
  const gsl_multimin_fminimizer_type *algorithm = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_function func;

  /* Starting point */
  gsl_vector* x = gsl_vector_alloc (dimension);
	auto v = generate_uniform_sample(rng, boundaries);
	for (size_t i = 0; i < dimension; i++) {
		gsl_vector_set(x, i, v[i]);
	}

  /* Set initial step sizes to 1 */
  gsl_vector* ss = gsl_vector_alloc (dimension);
  gsl_vector_set_all (ss, 1.0);

  /* Initialize method and iterate */
  func.n = dimension;
  func.f = &EGO::expected_improvement_bounded;
  func.params = this;

  gsl_multimin_fminimizer* s = gsl_multimin_fminimizer_alloc (algorithm, dimension);
  gsl_multimin_fminimizer_set (s, &func, x, ss);

	for (size_t trial = 0; trial < max_trials; trial++) {
      int status = gsl_multimin_fminimizer_iterate(s);

      if (status)
        break;

      double size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, convergence_threshold);

			if (status != GSL_CONTINUE) {
				break;
			}
	}

	vector<double> result;

	if (s->fval < 0.0) {
		for (size_t i = 0; i < dimension; i++) {
			result.push_back(gsl_vector_get(s->x, i));
		}
		improvement = -s->fval;
	}

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

	auto pair = ego->sg->predict(&x[0]);
	// Negate to maximise expected_improvement because this function is called by
	// a minimiser
	auto expectation = -expected_improvement(pair.first, pair.second, ego->best_fitness);

	if (expectation == 0.0 || !is_bounded(x, ego->boundaries)) {
		return euclidean_distance(x, ego->best_particle);
	}

	return expectation;
}

double EGO::expected_improvement(double y, double var, double y_min) {
	if (var <= 0.0) {
		return 0.0;
	}

	double sd = sqrt(var);
	double y_diff = y_min - y;
	double y_diff_s = y_diff / sd;
	return y_diff * gsl_cdf_ugaussian_P(y_diff_s) + sd * gsl_ran_ugaussian_pdf(y_diff_s);
}

/* Evaluates a vector to add to the training set */
void EGO::thread_evaluate(EGO* ego, vector<double> x) {
	assert(ego != NULL);

	vector<double> y = ego->evaluator.evaluate(x);

	ego->evaluator_lock.lock();

  ego->evaluations++;

	ego->sg->add(x, y[0], y[1] == 0 ? 1 : 2, 0);
	ego->sg_cost->add(x, y[2]);

	if (y[1] == 0) {
		if (y[0] < ego->best_fitness) {
			ego->best_particle = x;
			ego->best_fitness = y[0];
		}
	}

	ego->evaluator_lock.unlock();
}

/* Concurrently evaluates multiple points xs */
void EGO::evaluate(vector<vector<double>> xs) {
	vector<thread> threads;

	for (auto x : xs) {
		threads.push_back(thread(thread_evaluate, this, x));
	}

	for (auto& t : threads) {
		t.join();
	}
}
