#include <gsl/gsl_rng.h>
#include <utility>
#include <vector>

using namespace std;

typedef vector<pair<double, double>> boundaries_t;

bool is_bounded(vector<double> x, boundaries_t boundaries);

vector<double> generate_uniform_sample(gsl_rng* rng, boundaries_t boundaries);

double euclidean_distance(vector<double> x, vector<double> y);

void print_vector(vector<double> x);

double success_probability(double mean, double sd);

boundaries_t get_intersection(boundaries_t bxs, boundaries_t bys);

vector<vector<double>> generate_latin_samples(gsl_rng* rng, size_t samples,
  boundaries_t boundaries);

boundaries_t infer_boundaries(
    vector<pair<vector<double>, vector<double>>> results);

double apply_polynomial(double x, vector<double> coeffs);

bool is_subset(boundaries_t bxs, boundaries_t bys);
