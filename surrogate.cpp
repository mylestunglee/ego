#include "surrogate.hpp"
#include "gp.h"
#include "cg.h"
#include <thread>
#include <iostream>
#include <sstream>

using namespace std;
using namespace libgp;

// Surrogate is a wrapper for Gaussian processes
//   dimensions: number of dimensions in domain prediction space
//   log_transform: use log space for predictions
//   fixed_space: restrict space optimisation for performance
Surrogate::Surrogate(size_t dimension, bool log_transform, bool fixed_space) :
	dimension(dimension), log_transform(log_transform),
	fixed_space(fixed_space), trained(true), gp(newGaussianProcess()) {}

Surrogate::~Surrogate() {
	delete gp;
}

GaussianProcess* Surrogate::newGaussianProcess() {
	GaussianProcess* gp = new GaussianProcess(dimension, "CovSEard");
	Eigen::VectorXd params(gp->covf().get_param_dim());
	for(size_t i = 0; i < gp->covf().get_param_dim(); i++) {
		params(i) = 1;
	}
	gp->covf().set_loghyper(params);
	return gp;
}

void Surrogate::add(vector<double> x, double y) {
	assert(x.size() == dimension);
	assert(!log_transform || y > 0.0);
	gp->add_pattern(&x[0], log_transform ? log(y) : y);
	if (!fixed_space) {
		added.insert(make_pair(x, y));
	}
	trained = false;
}

// Optimise hyper-parameters
void Surrogate::train() {
	if (trained) {
		return;
	}

	CG cg;
	cg.maximize(gp, 50, false);
	trained = true;
}

// Get standard deviation of predictoin
double Surrogate::sd(vector<double> x) {
	assert(x.size() == dimension);
	return sqrt(max(gp->var(&x[0]), 0.0));
}

// Get mean of prediction
double Surrogate::mean(vector<double> x) {
	assert(x.size() == dimension);
	double mean = gp->f(&x[0]);
	return log_transform ? exp(mean) : mean;
}

// Switch between non-log and log space
void Surrogate::optimise_space() {
	assert(!fixed_space && !added.empty());
	GaussianProcess* gp_new = newGaussianProcess();
	for (auto pair : added) {
		gp_new->add_pattern(&pair.first[0],
			log_transform ? pair.second : log(pair.second));
	}
	if (gp->log_likelihood() > gp_new->log_likelihood()) {
		delete gp_new;
	} else {
		delete gp;
		gp = gp_new;
		log_transform = !log_transform;
	}
}

// Cross validation
double Surrogate::cross_validate() {
	assert(!fixed_space);
	vector<double> errors;

	// Select a point not to add
	for (auto pair : added) {
		Surrogate surrogate(dimension, false, false);
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
