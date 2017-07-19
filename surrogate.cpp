#include "surrogate.hpp"
#include "gp.h"
#include "cg.h"
#include <thread>
#include <iostream>
#include <sstream>

using namespace std;
using namespace libgp;

Surrogate::Surrogate(size_t dimension) {
  this->dimension = dimension;
  gp = new GaussianProcess(dimension, "CovSEard");

  Eigen::VectorXd params(gp->covf().get_param_dim());
  for(size_t i = 0; i < gp->covf().get_param_dim(); i++) {
    params(i) = 1;
  }
  gp->covf().set_loghyper(params);
}

Surrogate::~Surrogate() {
  delete gp;
}

void Surrogate::add(vector<double> x, double y) {
  assert(x.size() == dimension);
  gp->add_pattern(&x[0], y);
}

void Surrogate::train() {
  CG cg;
  cg.maximize(gp, 50, false);
}

double Surrogate::sd(vector<double> x) {
  assert(x.size() == dimension);
  double variance = gp->var(&x[0]);
  if (variance < 0.0) {
    return 0.0;
  }

  return sqrt(variance);
}

double Surrogate::mean(vector<double> x) {
  assert(x.size() == dimension);
  return gp->f(&x[0]);
}
