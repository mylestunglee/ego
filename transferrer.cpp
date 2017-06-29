#include "transferrer.hpp"
#include "csv.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <assert.h>
#include <set>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit.h>

using namespace std;

Transferrer::Transferrer(
	string filename_results_old,
	string filename_script_new,
	double sig_level
) : sig_level(sig_level) {
	read_results(filename_results_old);
	sort(results_old.begin(), results_old.end(), fitness_more_than);
	evaluator = new Evaluator(filename_script_new);
	boundaries = {make_pair(-1, 1), make_pair(-1, 1)};
}

Transferrer::~Transferrer() {
	delete evaluator;
}

/* Performs the automatic knowledge transfer */
void Transferrer::transfer() {

	vector<double> f_old;
	vector<double> f_new;
	for (auto result_old : sample_results_old()) {
		auto x_old = result_old.first;
		auto y_old = result_old.second;
		auto y_new = evaluator->evaluate(x_old);

		f_old.push_back(y_old[0]);
		f_new.push_back(y_new[0]);
	}

	double pearson;
	double spearman;
	calc_correlation(f_old, f_new, pearson, spearman);

	cout << "Pearson: " << pearson << ", Spearman: " << spearman << endl;

	// If hypothesis test for linear relationship between f_old and f_new passes
	if (1.0 - pearson < sig_level) {
		// Predict y_new_max
		double y_old_best = (results_old[0].second)[0];
		vector<double> coeff = fit_polynomial(f_old, f_new, 1);
		double y_new_best_approx = coeff[1] + coeff[0] * y_old_best;
		cout << "Old best: " << y_old_best << endl;
		cout << coeff[0] << endl << coeff[1] << endl;
		cout << "Predicted new best: " << y_new_best_approx << endl;
	}
}

/* Reads results from a CSV file from a previous EGO computation */
void Transferrer::read_results(string filename) {
	vector<vector<string>> data = read(filename);

	for (vector<string> line : data) {
		vector<double> x;
		vector<double> y;
		for (size_t i = 0; i < line.size(); i++) {
			double cell = atof(line[i].c_str());
			if (i < line.size() - 3) {
				x.push_back(cell);
			} else {
				y.push_back(cell);
			}
		}
		results_old.push_back(make_pair(x, y));
	}
}

/* Returns true iff x is bounded by boundaries_new */
bool Transferrer::is_bound(vector<double> x) {
	assert (x.size() == boundaries.size());

	for (size_t i = 0; i < x.size(); i++) {
		if (x[i] < boundaries[i].first || x[i] > boundaries[i].second) {
			return false;
		}
	}
	return true;
}

/* Calculates Pearson and Spearman correlation coefficents */
void Transferrer::calc_correlation(vector<double> x, vector<double> y, double &pearson, double& spearman) {
	assert(x.size() == y.size());

	size_t n = x.size();
		const size_t stride = 1;
	gsl_vector_const_view gsl_x = gsl_vector_const_view_array(&x[0], n);
	gsl_vector_const_view gsl_y = gsl_vector_const_view_array(&y[0], n);
	pearson = gsl_stats_correlation(
		(double*) gsl_x.vector.data, stride,
		(double*) gsl_y.vector.data, stride, n);
	double work[2 * n];
	spearman = gsl_stats_spearman(
		(double*) gsl_x.vector.data, stride,
		(double*) gsl_y.vector.data, stride, n, work);
}

/* Selects a subset of old_results to determine relationship of old and new evaluators */
vector<pair<vector<double>, vector<double>>> Transferrer::sample_results_old() {
	vector<pair<vector<double>, vector<double>>> result;
	set<vector<double>> sampled;
	const int trials = 50;

	for (int i = 0; i < trials; i++) {
		auto sample = results_old[rand() % results_old.size()];

		// Only add sample when sample fits new parameter space and has not been
		// sampled before
		if (is_bound(sample.first) && sampled.find(sample.first) == sampled.end()) {
			sampled.insert(sample.first);
			result.push_back(sample);
		}
	}

	return result;
}

/* Fits a set of 2D points to a N-dimensional polynomial fit */
vector<double> Transferrer::fit_polynomial(vector<double> dx, vector<double> dy, int degree)
{
	assert(dx.size() == dy.size());

	gsl_multifit_linear_workspace *ws;
	gsl_matrix* cov;
	gsl_matrix* X;
	gsl_vector* y;
	gsl_vector* c;
	double chisq;

	int n = dx.size();
	X = gsl_matrix_alloc(n, degree);
	y = gsl_vector_alloc(n);
	c = gsl_vector_alloc(degree);
	cov = gsl_matrix_alloc(degree, degree);

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < degree; j++) {
			gsl_matrix_set(X, i, j, pow(dx[i], j));
		}
		gsl_vector_set(y, i, dy[i]);
	}

	ws = gsl_multifit_linear_alloc(n, degree);
	gsl_multifit_linear(X, y, c, cov, &chisq, ws);

	vector<double> coeffs;

	for (int i = 0; i < degree; i++) {
		coeffs.push_back(gsl_vector_get(c, i));
	}

	gsl_multifit_linear_free(ws);
	gsl_matrix_free(X);
	gsl_matrix_free(cov);
	gsl_vector_free(y);
	gsl_vector_free(c);

	return coeffs;
}

/* Auxiliary less-than function to sort results */
bool Transferrer::fitness_more_than(
	pair<vector<double>, vector<double>> x,
	pair<vector<double>, vector<double>> y
) {
	return (x.second)[0] < (y.second)[0];
}
