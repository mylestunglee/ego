#include "surrogate.hpp"
#include "functions.hpp"
#include "gp.h"
#include "cg.h"
#include <thread>
#include <iostream>
#include <sstream>

using namespace std;
using namespace libgp;

Surrogate::Surrogate(size_t dimension) :
	dimension(dimension), log_transform(false), gp(NULL) {}

Surrogate::~Surrogate() {
	if (gp != NULL) {
		delete gp;
	}
}

// Wrapper for Gaussian Process library constructor
GaussianProcess* Surrogate::newGaussianProcess() {
	GaussianProcess* gp = new GaussianProcess(dimension, "CovSEard");
	Eigen::VectorXd params(gp->covf().get_param_dim());
	for(size_t i = 0; i < gp->covf().get_param_dim(); i++) {
		params(i) = 1;
	}
	gp->covf().set_loghyper(params);
	return gp;
}

// Add to training set
void Surrogate::add(vector<double> x, double y) {
	assert(x.size() == dimension);
	// Predictor is now old-of-date
	if (gp != NULL) {
		delete gp;
		gp = NULL;
	}

	added.insert(make_pair(x, y));
}

// Construct a predictor
void Surrogate::train() {
	assert(!added.empty());

	if (gp != NULL) {
		delete gp;
	}

	gp = newGaussianProcess();

	auto ys = extract_ys();

	if (log_transform) {
		auto zs = log_vector(ys);
		if (zs.empty()) {
			log_transform = false;
		} else {
			ys = zs;
		}
	}

	added_mean = sample_mean(ys);
	added_sd = sample_sd(ys);

	for (auto pair : added) {
		auto y = log_transform ? log(pair.second) : pair.second;
		auto y_normalised = (y - added_mean) / added_sd;
		gp->add_pattern(&pair.first[0], y_normalised);
	}

	// Optimise hyper-parameters
	CG cg;
	cg.maximize(gp, 50, false);
}

// Get standard deviation of prediction
double Surrogate::sd(vector<double> x) {
	assert(x.size() == dimension);
	if (gp == NULL) {
		train();
	}
	double var = gp->var(&x[0]);
	return sqrt(max(var, 0.0)) * added_sd;
}

// Get mean of prediction
double Surrogate::mean(vector<double> x) {
	assert(x.size() == dimension);
	if (gp == NULL) {
		train();
	}
	double mean_raw = gp->f(&x[0]);
	double mean = log_transform ? exp(mean_raw) : mean_raw;
	return mean * added_sd + added_mean;
}

// Switch between non-log and log space
void Surrogate::optimise_space() {
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
double Surrogate::cross_validate() {
	vector<double> errors;

	// Select a point not to add
	for (auto pair : added) {
		Surrogate surrogate(dimension);
		for (auto add : added) {
			if (pair == add) {
				 continue;
			}
			surrogate.add(add.first, add.second);
		}
		surrogate.train();
		surrogate.optimise_space();

		auto x = pair.first;
		auto y = pair.second;
		double error = (y - surrogate.mean(x)) / y;
		errors.push_back(error);
	}
	return accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

// Extracts range from added points
vector<double> Surrogate::extract_ys() {
	vector<double> ys;
	for (auto pair : added) {
		ys.push_back(pair.second);
	}
	return ys;
}
