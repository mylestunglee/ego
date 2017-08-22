#include "animation.hpp"
#include "functions.hpp"
#include "gp.hpp"
#include "gp.h"
#include "cg.h"
#include <thread>
#include <iostream>
#include <sstream>
#include <limits>

using namespace std;

GaussianProcess::GaussianProcess(size_t dimension) :
	dimension(dimension), log_transform(false), gp(NULL),
	x_mean(dimension, 0.0), x_sd(dimension, 1.0), y_mean(0.0), y_sd(1.0) {}

GaussianProcess::~GaussianProcess() {
	if (gp != NULL) {
		delete gp;
	}
}

// Wrapper for Gaussian Process library constructor
libgp::GaussianProcess* GaussianProcess::newGaussianProcess() {
	auto gp = new libgp::GaussianProcess(dimension, "CovSEard");
	Eigen::VectorXd params(gp->covf().get_param_dim());
	for(size_t i = 0; i < gp->covf().get_param_dim(); i++) {
		params(i) = 1;
	}
	gp->covf().set_loghyper(params);
	return gp;
}

// Add to training set
void GaussianProcess::add(vector<double> x, double y) {
	assert(!isnan(y));

	assert(x.size() == dimension);
	// Predictor is now old-of-date
	if (gp != NULL) {
		delete gp;
		gp = NULL;
	}

	added.insert(make_pair(x, y));
}

// Construct a predictor
void GaussianProcess::train() {
	assert(!added.empty());

	if (gp != NULL) {
		delete gp;
	}

	gp = newGaussianProcess();

//	update_whitener();

	for (auto pair : added) {
		auto x = pair.first;
		auto x_normalised = normalise_x(x);
		auto y = log_transform ? log(pair.second) : pair.second;
		auto y_normalised = (y - y_mean) / y_sd;

		gp->add_pattern(&x_normalised[0], y_normalised);
	}

	// Optimise hyper-parameters
	libgp::CG cg;
	cg.maximize(gp, 50, false);
}

// Get standard deviation of prediction
double GaussianProcess::sd(vector<double> x) {
	assert(x.size() == dimension);
	if (gp == NULL) {
		train();
	}
	auto x_normalised = normalise_x(x);
	double var = gp->var(&x_normalised[0]);
	if (isnan(var)) {
		return 0.0;
	}

	return sqrt(max(var, 0.0)) * y_sd;
}

// Get mean of prediction
double GaussianProcess::mean(vector<double> x) {
	assert(x.size() == dimension);
	if (gp == NULL) {
		train();
	}
	auto x_normalised = normalise_x(x);
	double mean_raw = gp->f(&x_normalised[0]);
	if (isnan(mean_raw)) {
		mean_raw = 0.0;
	}
	double mean = mean_raw * y_sd + y_mean;
	return log_transform ? exp(mean) : mean;
}

// Switch between non-log and log space
void GaussianProcess::optimise() {
	assert(!added.empty());
	if (gp == NULL) {
		train();
	}

	auto gp_old = gp;

	// Attempt training with other space
	log_transform = !log_transform;
	gp = NULL;
	train();

	// Previous result was better
	if (gp_old->log_likelihood() > gp->log_likelihood()) {
		// Revert changes
		delete gp;
		gp = gp_old;
		log_transform = !log_transform;
	} else {
		delete gp_old;
	}
}

// Cross validation
double GaussianProcess::cross_validate() {
	if (added.size() <= 1) {
		return numeric_limits<double>::infinity();
	}

	animation_start("Cross validating", 0, added.size());
	vector<double> errors;

	// Select a point not to add
	for (auto pair : added) {
		GaussianProcess surrogate(dimension);
		for (auto add : added) {
			if (pair == add) {
				 continue;
			}
			surrogate.add(add.first, add.second);
		}
		auto x = pair.first;
		auto y = pair.second;
		double error = abs(y - surrogate.mean(x));
		errors.push_back(error);
		animation_step();
	}
	double mean = sample_mean(extract_ys());
	return accumulate(errors.begin(), errors.end(), 0.0) /
		((double)errors.size() * mean);
}

// Extracts ys from added points
vector<double> GaussianProcess::extract_ys() {
	vector<double> ys;
	for (auto pair : added) {
		ys.push_back(pair.second);
	}
	return ys;
}

// Extracts xs from added points
vector<vector<double>> GaussianProcess::extract_xs() {
	assert(!added.empty());
	vector<vector<double>> result(dimension, vector<double>());
	for (auto pair : added) {
		auto x = pair.first;
		for (size_t i = 0; i < dimension; i++) {
			result[i].push_back(x[i]);
		}
	}
	return result;
}

// Updates the linear transformation values to whiten input data
void GaussianProcess::update_whitener() {
	// Update xs
	auto xs = extract_xs();

	x_mean.clear();
	x_sd.clear();

	for (auto x : xs) {
		x_mean.push_back(sample_mean(x));
		x_sd.push_back(sample_sd(x));
	}

	// Update ys
	auto ys = extract_ys();

	if (log_transform) {
		auto zs = log_vector(ys);
		if (zs.empty()) {
			log_transform = false;
		} else {
			ys = zs;
		}
	}

	y_mean = sample_mean(ys);
	y_sd = sample_sd(ys);
}

// Apply linear transformations to whiten x
vector<double> GaussianProcess::normalise_x(vector<double> x) {
	assert(x.size() == dimension);
	vector<double> normalised;
	for (size_t i = 0; i < dimension; i++) {
		normalised.push_back((x[i] - x_mean[i]) / x_sd[i]);
	}
	return normalised;
}
