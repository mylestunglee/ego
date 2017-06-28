
// Pearson and spearman correlation examples

#include "transfer.hpp"
#include <iostream>
#include <vector>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_vector.h>

using namespace std;

void transfer() {

}

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
