#include <assert.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include "functions.hpp"

// Returns true iff x is bounded inside an n-dimensional hypercube defined by boundaries
bool is_bounded(vector<double> x, vector<pair<double, double>> boundaries) {
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
vector<double> generate_uniform_sample(gsl_rng* rng, vector<pair<double, double>> boundaries) {
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
