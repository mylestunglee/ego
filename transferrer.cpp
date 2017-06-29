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

using namespace std;

Transferrer::Transferrer(string filename_results_old, string filename_script_new)
{
	read_results(filename_results_old);
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
		old_results.push_back(make_pair(x, y));
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
		auto sample = old_results[rand() % old_results.size()];

		// Only add sample when sample fits new parameter space and has not been
		// sampled before
		if (is_bound(sample.first) && sampled.find(sample.first) == sampled.end()) {
			sampled.insert(sample.first);
			result.push_back(sample);
		}
	}

	return result;
}
