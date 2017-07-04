#include <gsl/gsl_rng.h>
#include <utility>
#include <vector>

using namespace std;

bool is_bounded(vector<double> x, vector<pair<double, double>> boundaries);

vector<double> generate_uniform_sample(gsl_rng* rng, vector<pair<double, double>> boundaries);
