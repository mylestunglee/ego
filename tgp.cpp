#include <utility>
#include "tgp.hpp"
#include "functions.hpp"

using namespace std;

TransferredGaussianProcess::TransferredGaussianProcess(
	set<pair<vector<double>, double>> added) :
	added_old(added), transferred(NULL), parameter(NULL) {}

TransferredGaussianProcess::~TransferredGaussianProcess() {
	if (transferred != NULL) {
		delete transferred;
		delete parameter;
	}
}

void TransferredGaussianProcess::add(vector<double> x, double y) {
	if (transferred != NULL) {
		delete transferred;
		delete parameter;
		transferred = NULL;
		parameter = NULL;
	}
	added_new.insert(make_pair(x, y));
}

double TransferredGaussianProcess::mean(vector<double> x) {
	if (transferred == NULL) {
		train();
	}
	return transferred->mean(x);
}

double TransferredGaussianProcess::sd(vector<double> x) {
	if (transferred == NULL) {
		train();
	}
	return transferred->mean(x);
}

void TransferredGaussianProcess::optimise() {
	if (transferred == NULL) {
		train();
	}
	transferred->optimise();
	train();
}

double TransferredGaussianProcess::cross_validate() {
	if (transferred == NULL) {
		train();
	}
	return transferred->cross_validate();
}

double TransferredGaussianProcess::cross_validate_parameter() {
	if (transferred == NULL) {
		train();
	}
	return parameter->cross_validate();
}

// Construct predictor of added_new using added_old
void TransferredGaussianProcess::train() {
	assert(!added_old.empty());

	// Reconstruct Gaussian processes
	size_t dimension = added_old.begin()->first.size();
	transferred = new GaussianProcess(dimension);
	parameter = new GaussianProcess(dimension);

	// Train without knowledge transfer
	for (auto pair : added_new) {
		transferred->add(pair.first, pair.second);
	}

	// Train with knowledge transfer
	vector<pair<vector<double>, vector<double>>> xs;
	set<pair<vector<double>, double>> flexible;

	for (auto pair_old : added_old) {
		bool found = false;
		auto x = pair_old.first;
		double y_old = pair_old.second;
		double y_new;
		// Find fitness mapping
		for (auto pair_new : added_new) {
			if (x == pair_new.first) {
				found = true;
				y_new = pair_new.second;
				break;
			}
		}
		if (found) {
			xs.push_back(make_pair(x, vector<double>()));
			parameter->add(x, transfer_calc_parameter(y_old, y_new));
		} else {
			flexible.insert(pair_old);
		}
	}

	boundaries_t boundaries = infer_boundaries(xs);

	for (auto pair : flexible) {
		auto x = pair.first;
		// Do not transfer points outside of prediction boundaries
		if (!is_bounded(x, boundaries)) {
			continue;
		}
		auto y = pair.second;

		// Attempt to transfer
		auto p = parameter->mean(x);
		if (!isnan(p)) {
			transferred->add(x, transfer_fitness_predict(y, p));
		}
	}
}
