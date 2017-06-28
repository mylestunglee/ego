#include "transfer.hpp"
#include "csv.hpp"
#include "evaluator.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>

using namespace std;

/* Reads results from a CSV file from a previous EGO computation */
vector<pair<vector<double>, vector<double>>> read_results(string filename) {
	vector<vector<string>> data = read(filename);
	vector<pair<vector<double>, vector<double>>> result;

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
		result.push_back(make_pair(x, y));
	}

	return result;
}

/* Returns true iff for each ith dimension of x, x_i is between lower_i and upper_i */
bool is_bound(vector<double> x, vector<double> lower, vector<double> upper) {
	for (size_t i = 0; i < x.size(); i++) {
		if (x[i] < lower[i] || x[i] > upper[i]) {
			return false;
		}
	}
	return true;
}

void transfer(string filename_results_old, string filename_script_new) {
	auto results_old = read_results(filename_results_old);

	random_shuffle(results_old.begin(), results_old.end());

	vector<double> lower = {-1, -1};
	vector<double> upper = {1, 1};

	Evaluator evaluator(filename_script_new);

	for (auto result_old : results_old) {
		auto x_old = result_old.first;
		auto y_old = result_old.second;
		if (is_bound(x_old, lower, upper)) {
			auto y_new = evaluator.evaluate(x_old);
	for (size_t i = 0; i < y_old.size(); i++) {
		cout << "i = " << i << ", y_old_i = " << y_old[i] << ", y_new_i = " << y_new[i] << endl;
	}

			break;
		}
	}
}

// Pearson and spearman correlation example
void test_lib() {
  vector<double> x = {1, 3, 5, 7, 9};
  vector<double> y = {2, 5, 5, 4, 9};
  size_t n = x.size();
  const size_t stride = 1;
  gsl_vector_const_view gsl_x = gsl_vector_const_view_array(&x[0], n);
  gsl_vector_const_view gsl_y = gsl_vector_const_view_array(&y[0], n);
  double pearson = gsl_stats_correlation(
    (double*) gsl_x.vector.data, stride,
    (double*) gsl_y.vector.data, stride, n);
  cout << "Pearson correlation: " << pearson << endl;
  double work[2 * n];
  double spearman = gsl_stats_spearman(
    (double*) gsl_x.vector.data, stride,
    (double*) gsl_y.vector.data, stride, n, work);
  cout << "Spearman correlation: " << spearman << endl;
}
