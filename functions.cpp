#include <assert.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <gsl_randist.h>
#include <gsl_cdf.h>
#include "ihs.hpp"
#include "functions.hpp"

// Returns true iff x is bounded inside an n-dimensional hypercube defined by boundaries
bool is_bounded(vector<double> x, boundaries_t boundaries) {
	assert(x.size() == boundaries.size());

	for (size_t i = 0; i < x.size(); i++) {
		assert(boundaries[i].first <= boundaries[i].second);
		if (x[i] < boundaries[i].first || x[i] > boundaries[i].second) {
			return false;
		}
	}

	return true;
}

// Returns a sample from a uniform distribution of a hypercube defind by boundaries
vector<double> generate_uniform_sample(gsl_rng* rng, boundaries_t boundaries) {
	vector<double> sample;

	for (auto boundary : boundaries) {
		assert(boundary.first <= boundary.second);
		sample.push_back(gsl_ran_flat(rng, boundary.first, boundary.second));
	}

	return sample;
}

// Returns the Euclidean distance between two points
double euclidean_distance(vector<double> x, vector<double> y) {
	assert (x.size() == y.size());
	double distance = 0.0;
	for (size_t i = 0; i < x.size(); i++) {
		distance += pow(x[i] - y[i], 2);
	}
	return sqrt(distance);
}

// Pretty prints a vector
void print_vector(vector<double> x) {
	cout << "(" << setprecision(3);
	for (size_t i = 0; i < x.size(); i++) {
		cout << x[i];
		if (i < x.size() - 1) {
			cout << ",";
		}
	}
	cout << ")";
}

// Finds the probability of success given a normal distribution
double success_probability(double mean, double sd) {
	double upper = gsl_cdf_gaussian_P(1.5 - mean, sd);
	double middle = gsl_cdf_gaussian_P(0.5 - mean, sd);
	double lower = gsl_cdf_gaussian_P(-0.5 - mean, sd);
	double success = upper - middle;
	double normal = upper - lower;
	return success / normal;
}

// Returns the hypercube intersection of the two hypercubes bxs and bys
boundaries_t get_intersection(boundaries_t bxs, boundaries_t bys) {
	assert(bxs.size() == bys.size());
	boundaries_t result;
	for (size_t i = 0; i < bxs.size(); i++) {
		double lower = min(bxs[i].first, bys[i].first);
		double upper = max(bxs[i].second, bys[i].second);
		if (lower > upper) {
			return {};
		} else {
			result.push_back(make_pair(lower, upper));
		}
	}
	return result;
}

// Generates a Latin hypercube bounded by boundaries with samples samples
vector<vector<double>> generate_latin_samples(gsl_rng* rng, size_t samples,
	boundaries_t boundaries) {
	size_t dimension = boundaries.size();
	int seed = gsl_rng_get(rng);
	int* latin = ihs(dimension, samples, 5, seed);
	assert(latin != NULL);

	vector<vector<double>> xs;

	// Scale latin hypercube to fit parameter space
	for (size_t i = 0; i < samples; i++) {
		vector<double> x;
		for (size_t j = 0; j < dimension; j++) {
			double lower = boundaries[j].first;
			double upper = boundaries[j].second;
			double x_j = lower + (latin[i * dimension + j] - 1.0) /
				(samples - 1.0) * (upper - lower);
			x.push_back(x_j);
		}
		xs.push_back(x);
	}

	delete latin;

	return xs;
}

// Returns the boundaries that bounds all xs in results
boundaries_t infer_boundaries(
    vector<pair<vector<double>, vector<double>>> results) {
	assert(!results.empty());

	vector<vector<double>> xs(results[0].first.size(), vector<double>());
	for (auto result : results) {
		auto x = result.first;
		for (size_t i = 0; i < x.size(); i++) {
			xs[i].push_back(x[i]);
		}
	}

	boundaries_t boundaries;
	for (auto x : xs) {
		auto boundary = minmax_element(x.begin(), x.end());
		boundaries.push_back(make_pair(x[boundary.first - x.begin()],
			x[boundary.second - x.begin()]));
	}

	return boundaries;
}

// Given a polynomial of coefficents c_0, c_1..., compute c_0 + c_1x + c_2x^2...
double apply_polynomial(double x, vector<double> coeffs) {
	double sum = 0.0;
	for (size_t i = 0; i < coeffs.size(); i++) {
		sum += coeffs[i] * pow(x, i);
	}
	return sum;
}

// Returns true iff all points in bxs are bounded by bys
bool is_subset(boundaries_t bxs, boundaries_t bys) {
	assert(bxs.size() == bys.size());
	for (size_t i = 0; i < bxs.size(); i++) {
		if (bxs[i].first < bys[i].first || bxs[i].second > bys[i].second) {
			return false;
		}
	}
	return true;
}
