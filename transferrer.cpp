#include "transferrer.hpp"
#include "csv.hpp"
#include "constants.hpp"
#include "surrogate.hpp"
#include "functions.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <assert.h>
#include <set>
#include <gsl_statistics.h>
#include <gsl_vector.h>
#include <gsl_multifit.h>
#include <gsl_cdf.h>

using namespace std;

Transferrer::Transferrer(
	string filename_results_old,
	string filename_script_new,
	double sig_level
) : sig_level(sig_level) {
	read_results(filename_results_old);
	sort(results_old.begin(), results_old.end(), fitness_more_than);
	evaluator = new Evaluator(filename_script_new);
	boundaries = {make_pair(-1, 1), make_pair(-1, 3)};
}

Transferrer::~Transferrer() {
	delete evaluator;
}

/* Performs the automatic knowledge transfer */
void Transferrer::transfer() {
	vector<double> ys_old;
	vector<double> ys_new;
	vector<pair<vector<double>, int>> sample_labels_new;

	auto sample = sample_results_old();

	// Compute fitness for sample
	for (auto result_old : sample) {
		auto x_old = result_old.first;
		auto y_old = result_old.second;
		auto y_new = evaluator->evaluate(x_old);

		ys_old.push_back(y_old[FITNESS_INDEX]);
		ys_new.push_back(y_new[FITNESS_INDEX]);
		sample_labels_new.push_back(make_pair(x_old, y_new[LABEL_INDEX]));
	}

	double label_correlation = calc_label_correlation(sample_labels_new);
	cout << "Label correlation: " << label_correlation << endl;

	if (1.0 - label_correlation > sig_level) {
		cout << "Insufficent consistency of labels." << endl;
		return;
	}

	double pearson;
	double spearman;
	calc_correlation(ys_old, ys_new, pearson, spearman);

	cout << "Pearson: " << pearson << ", Spearman: " << spearman << endl;
	double y_old_best = (results_old[0].second)[FITNESS_INDEX];
	vector<double> coeffs;

	// If hypothesis test for linear relationship between ys_old and ys_new passes
	if (1.0 - pearson <= sig_level || 1.0 + pearson <= sig_level) {
		coeffs = fit_polynomial(ys_old, ys_new, 1);
		cout << "Using linear regression" << endl;

	// If hypothesis test for monotonic relationship passes
	} else if (1.0 - spearman <= sig_level || 1.0 + spearman <= sig_level) {
		coeffs = fit_polynomial(ys_old, ys_new, 2);
		cout << "Using quadratic regression" << endl;
	} else {
		cout << "Insufficent monotonic relationship between fitness functions." << endl;
		return;
	}

	double y_new_best_approx = apply_polynomial(y_old_best, coeffs);
	boundaries_t boundaries_old = infer_boundaries(results_old);

	if (is_subset(boundaries, boundaries_old)) {
		cout << "Trivial mapping, y_new_opt = " << y_new_best_approx << endl;
		return;
	}

	boundaries_t intersection = get_intersection(boundaries_old, boundaries);

	for (auto boundary_old : boundaries) {
		cout << boundary_old.first << ", " << boundary_old.second << endl;
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

/* Calculates Pearson and Spearman correlation coefficents */
void Transferrer::calc_correlation(vector<double> xs, vector<double> ys, double &pearson, double& spearman) {
	assert(xs.size() == ys.size());

	size_t n = xs.size();
		const size_t stride = 1;
	gsl_vector_const_view gsl_x = gsl_vector_const_view_array(&xs[0], n);
	gsl_vector_const_view gsl_y = gsl_vector_const_view_array(&ys[0], n);
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

	for (unsigned i = 0; i < SAMPLE_TRIALS; i++) {
		auto sample = results_old[rand() % results_old.size()];

		// Only add sample when sample fits new parameter space and has not been
		// sampled before
		if (is_bounded(sample.first, boundaries) &&
			sampled.find(sample.first) == sampled.end()) {
			sampled.insert(sample.first);
			result.push_back(sample);
		}

		if (sampled.size() > SAMPLE_MAX) {
			break;
		}
	}

	return result;
}

/* Fits a set of 2D points to a N-dimensional polynomial fit */
vector<double> Transferrer::fit_polynomial(vector<double> xs, vector<double> ys, int degree)
{
	assert(xs.size() == ys.size());

	gsl_multifit_linear_workspace *ws;
	gsl_matrix* cov;
	gsl_matrix* X;
	gsl_vector* y;
	gsl_vector* c;
	double chisq;
	int order = degree + 1;
	int n = xs.size();
	X = gsl_matrix_alloc(n, order);
	y = gsl_vector_alloc(n);
	c = gsl_vector_alloc(order);
	cov = gsl_matrix_alloc(order, order);

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < order; j++) {
			gsl_matrix_set(X, i, j, pow(xs[i], j));
		}
		gsl_vector_set(y, i, ys[i]);
	}

	ws = gsl_multifit_linear_alloc(n, order);
	gsl_multifit_linear(X, y, c, cov, &chisq, ws);

	vector<double> coeffs;

	for (int i = 0; i < order; i++) {
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

/* Calculates the average p-value from confidence interval tests on all samples */
double Transferrer::calc_label_correlation(vector<pair<vector<double>, int>> sample_labels) {
	assert(!sample_labels.empty());

	Surrogate surrogate(boundaries.size());

	// Converts labels into continuous values
	for (auto result_old : results_old) {
		surrogate.add(
			result_old.first,
			(result_old.second)[LABEL_INDEX] == 0.0 ? 0.0 : 1.0);
	}
	surrogate.train();

	vector<double> coeffs;

	for (auto sample_label : sample_labels) {
		auto x = sample_label.first;
		double mean = surrogate.mean(x);
		double sd = surrogate.sd(x);

		// If SVM prediction is certain, otherwise assume normal distribution
		if (isnan(sd)) {
			coeffs.push_back(1.0);
		} else if (sample_label.second == 0.0) {
			coeffs.push_back(gsl_cdf_gaussian_P(1.0 - mean, sd));
		} else {
			coeffs.push_back(gsl_cdf_gaussian_Q(-mean, sd));
		}
	}

	return accumulate(coeffs.begin(), coeffs.end(), 0.0) / coeffs.size();
}
