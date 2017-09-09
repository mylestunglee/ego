#pragma once
#include "gp.hpp"
#include <gsl_rng.h>
#include <gsl_vector.h>
#include <utility>
#include <vector>
#include <functional>

using namespace std;

typedef vector<pair<double, double>> boundaries_t;
typedef vector<pair<vector<double>, vector<double>>> results_t;

struct config {
	size_t max_evaluations;
	size_t max_trials;
	size_t constraints;
	size_t costs;
	double convergence_threshold;
	double sig_level;
	double fitness_percentile;
	bool is_discrete;
	boundaries_t boundaries;
	vector<string> names;
	vector<string> tags;
};

typedef struct config config_t;

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

void calc_correlation(vector<double> xs, vector<double> ys,
	double &pearson, double& spearman);

bool are_valid_boundaries(boundaries_t boundaries);

double calc_hypervolume(boundaries_t boundaries);

vector<pair<double, double>> read_boundaries(vector<string> xs, vector<string> ys);

vector<double> round_vector(vector<double> x);

vector<vector<double>> generate_grid_samples(size_t density, boundaries_t boundaries);

vector<double> minimise_local(double (*func)(const gsl_vector*, void*),
	void* arg, vector<double> x, double convergence_threshold,
	size_t max_trials, double& minimum);

vector<double> minimise(double (*func)(const gsl_vector*, void*),
	vector<double> (*gen)(void*), void* arg, double convergence_threshold,
	size_t max_trials, function<bool (vector<double> x)> pred, double& minimum);

vector<double> gsl_to_std_vector(const gsl_vector* v);

void print_boundaries(boundaries_t boundaries);

double transfer_fitness_predict(double fitness_old, double parameter);

double transfer_calc_parameter(double fitness_old, double fitness_new);

void log_surrogate_predictions(Surrogate& surrogate, string filename, boundaries_t boundaries);

vector<vector<double>> generate_all_samples(boundaries_t boundaries);

void log_fitness(double fitness);

void write_fitness_log(string filename);

vector<double> calc_midpoint(vector<vector<double>> xs);

vector<double> multilinear_regression_fit(vector<vector<double>> xs, vector<double> ys);

vector<vector<double>> multiquadratic_regression_fit(vector<vector<double>> xs,
	vector<double> ys);

vector<vector<double>> multiquadratic_result_extrapolate(results_t results);

void log_multiquadratic_extrapolation(vector<vector<double>> fs,
	string filename, boundaries_t boundaries, boundaries_t rejection);

vector<double> minimise_multiquadratic(vector<vector<double>> fs,
	boundaries_t boundaries);

vector<vector<double>> generate_sparse_latin_samples(gsl_rng* rng,
	vector<vector<double>> xs, size_t samples, size_t max_trials,
	boundaries_t boundaries);

boundaries_t prune_boundaries(boundaries_t boundaries,
	boundaries_t boundaries_old, vector<vector<double>> quadratics,
	vector<double> correlations, double sig_level);

vector<double> calc_spearmans(results_t results);

double sample_mean(vector<double> xs);

double sample_sd(vector<double> xs);

vector<double> log_vector(vector<double> xs);

results_t read_results(string filename, size_t dimension);

size_t count_common_results(vector<results_t> results_olds,
	results_t results_new);

size_t count_common_results(vector<results_t> resultss);

void add_results_to_surrogate(results_t& results, Surrogate& surrogate);

vector<vector<double>> calc_cluster_midpoints(vector<results_t> results,
	size_t n);

vector<double> extract_cluster_midpoint(vector<results_t>& results);

void extract_cluster_midpoint_auxiliary(vector<results_t>& results,
	size_t step, vector<size_t>& indices, double& best_distance,
	vector<double>& best_midpoint, vector<size_t>& best_indices);

bool read_config(string filename, config_t& config);
