#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <utility>
#include <vector>

using namespace std;

typedef vector<pair<double, double>> boundaries_t;
typedef vector<pair<vector<double>, vector<double>>> results_t;

bool is_bounded(vector<double> x, boundaries_t boundaries);

vector<double> generate_uniform_sample(gsl_rng* rng, boundaries_t boundaries);

double euclidean_distance(vector<double> x, vector<double> y);

void print_vector(vector<double> x);

double success_probability(double mean, double sd);

boundaries_t get_intersection(boundaries_t bxs, boundaries_t bys);

vector<vector<double>> generate_latin_samples(gsl_rng* rng, size_t samples,
  boundaries_t boundaries);

boundaries_t infer_boundaries(results_t results);

double apply_polynomial(double x, vector<double> coeffs);

bool is_subset(boundaries_t bxs, boundaries_t bys);

vector<double> fit_polynomial(vector<double> xs, vector<double> ys, int degree);

void calc_correlation(vector<double> xs, vector<double> ys,
	double &pearson, double& spearman);

bool are_valid_boundaries(boundaries_t boundaries);

double calc_hypervolume(boundaries_t boundaries);

vector<pair<double, double>> read_boundaries(vector<string> xs, vector<string> ys);

vector<double> round_vector(vector<double> x);

vector<vector<double>> generate_grid_samples(size_t density, boundaries_t boundaries);

vector<double> join_vectors(vector<double> x, vector<double> y);

boundaries_t join_boundaries(boundaries_t x, boundaries_t y);

vector<double> minimise_local(double (*func)(const gsl_vector*, void*),
	void* arg, vector<double> x, double convergence_threshold,
	size_t max_trials, double& minimum);

vector<double> minimise(double (*func)(const gsl_vector*, void*),
	vector<double> (*gen)(void*), void* arg, double convergence_threshold,
	size_t max_trials, double& minimum);

bool is_success(vector<double> y, size_t constraints, size_t costs);

vector<double> gsl_to_std_vector(const gsl_vector* v);

void print_boundaries(boundaries_t boundaries);
