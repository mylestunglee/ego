#include <limits>
#include "ego.hpp"
#include "ihs.hpp"
#include <ctime>
#include <thread>
#include <chrono>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_multimin.h>

using namespace std;

EGO::EGO(vector<pair<double, double>> boundaries, Evaluator& evaluator) :
	evaluator(evaluator)
{
  dimension = boundaries.size();

	for (auto boundary : boundaries) {
		lower.push_back(boundary.first);
		upper.push_back(boundary.second);
	}

	sg = new Surrogate(boundaries.size(), SEiso, true, false);
	sg_cost = new Surrogate(boundaries.size(), SEard);

	rng = gsl_rng_alloc(gsl_rng_taus);

  max_iterations = 1000;
  num_iterations = 0;
  best_fitness = 10000000000;
  is_discrete = false;
}

EGO::~EGO() {
	gsl_rng_free(rng);

	delete sg;
	delete sg_cost;
}

void EGO::run_quad()
{
	assert(num_iterations > 0);

	sg->train_gp_first();
	while(num_iterations < max_iterations) {
		sg->train();
		sg_cost->train();

		cout << "Iteration: " << num_iterations << endl;
		evaluate({maximise_expected_improvement()});
		cout << "Best @= " << best_particle[0] << ", " << best_particle[1] << endl;
	}
}

vector<double> EGO::best_result()
{
  return best_particle;
}

void EGO::sample_plan(size_t n)
{
	int seed = gsl_rng_get(rng);
	int* latin = ihs(dimension, n, 5, seed);
	assert(latin != NULL);

	vector<vector<double>> xs;

	// Scale latin hypercube to fit parameter space
	for (size_t i = 0; i < n; i++) {
		vector<double> x;
		for (size_t j = 0; j < (unsigned) dimension; j++) {
			double x_j = lower[j] + (latin[i * dimension + j] - 1.0) / (n - 1.0) * (upper[j] - lower[j]);
			if (is_discrete) {
				x_j = round(x_j);
			}
			x.push_back(x_j);
		}
		xs.push_back(x);
	}

	delete latin;

	evaluate(xs);

	sg->choose_svm_param(5, true);
}

void EGO::uniform_sample(size_t n) {
	for (size_t i = 0; i < n; i++) {
		vector<double> x(dimension, 0.0);
		for (size_t trial = 0; trial < 30; trial++) {
			// Sample parameter space using uniform distribution
			for (int j = 0; j < dimension; j++) {
				x[j] = gsl_ran_flat(rng, lower[j], upper[j]);
			}

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

vector<double> EGO::maximise_expected_improvement() {
  const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer *s = NULL;
  gsl_vector *ss, *x;
  gsl_multimin_function minex_func;

  size_t iter = 0;
  int status;
  double size;

  /* Starting point */
  x = gsl_vector_alloc (dimension);

	for (int i = 0; i < dimension; i++) {
		gsl_vector_set(x, i, gsl_ran_flat(rng, lower[i], upper[i]));
	}

  /* Set initial step sizes to 1 */
  ss = gsl_vector_alloc (dimension);
  gsl_vector_set_all (ss, 1.0);

  /* Initialize method and iterate */
  minex_func.n = 2;
  minex_func.f = &EGO::expected_improvement;
  minex_func.params = this;

  s = gsl_multimin_fminimizer_alloc (T, dimension);
  gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

	vector<double> result;

  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);
      
      if (status) 
        break;

      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-2);

      if (status == GSL_SUCCESS)
        {
          printf ("converged to minimum at\n");
			for (int i = 0; i < dimension; i++) {
				result.push_back(gsl_vector_get(s->x, i));
			}
      printf ("%5d %10.3e %10.3e f() = %7.3f size = %.3f\n", 
              iter,
              gsl_vector_get (s->x, 0), 
              gsl_vector_get (s->x, 1), 
              s->fval, size);
        }

    }
  while (status == GSL_CONTINUE && iter < 100);

  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);

  return result;
}

double EGO::expected_improvement(const gsl_vector* v, void* p) {
	EGO* ego = (EGO*) p;

	vector<double> x;
	for (int i = 0; i < ego->dimension; i++) {
		x.push_back(gsl_vector_get(v, i));
	}

	assert(x.size() != 0);

	auto pair = ego->sg->predict(&x[0]);
	double e = -ei(pair.first, pair.second, ego->best_fitness);
	// if out of bounds
	for (int i = 0; i < ego->dimension; i++) {
		if (x[i] < ego->lower[i] || x[i] > ego->upper[i]) {
			e = 0;
			break;
		}
	}

	if (e == 0) {
		// v in some invalid region, just say it's some gradient towards the best fitness
		double distance = 0.0;
		for (int i = 0; i < ego->dimension; i++) {
			distance += pow(x[i] - ego->best_particle[i], 2);
		}

		return sqrt(distance) * 100;
	} else {
		// return expected improvement
		return e;
	}

}

double EGO::ei(double y, double var, double y_min) {
	if (var <= 0.0) {
		return 0.0;
	}

	double sd = sqrt(var);
	double y_diff = y_min - y;
	double y_diff_s = y_diff / sd;
	return y_diff * gsl_cdf_ugaussian_P(y_diff_s) + sd * gsl_ran_ugaussian_pdf(y_diff_s);
}

/* Evaluates a vector to add to the training set */
void EGO::evaluate2(EGO* ego, vector<double> x) {
	assert(ego != NULL);

	vector<double> y = ego->evaluator.evaluate(x);

	ego->running_mtx.lock();

    ego->num_iterations++;

	ego->sg->add(x, y[0], y[1] == 0 ? 1 : 2, 0);
	ego->sg_cost->add(x, y[2]);

	if (y[1] == 0) {
		if (y[0] < ego->best_fitness) {
			ego->best_particle = x;
			ego->best_fitness = y[0];
		}
	}

	ego->running_mtx.unlock();
}

/* Concurrently evaluates multiple points xs */
void EGO::evaluate(vector<vector<double>> xs) {
	vector<thread> threads;

	for (auto x : xs) {
		threads.push_back(thread(evaluate2, this, x));
	}

	for (auto& t : threads) {
		t.join();
	}
}

