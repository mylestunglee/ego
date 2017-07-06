#include "surrogate.hpp"
#include "gp.h"
#include "cg.h"
#include <thread>
#include <iostream>
#include <sstream>

using namespace std;
using namespace libgp;

Surrogate::Surrogate(int d, s_type type, bool svm, bool log_b)
{
  dim = d;

  gp = new GaussianProcess(dim, "CovSEard");
  Eigen::VectorXd params(gp->covf().get_param_dim());
  for(size_t i = 0; i < gp->covf().get_param_dim(); i++) {
    params(i) = -1;
  }
  gp->covf().set_loghyper(params);
}

void Surrogate::choose_svm_param(int num_folds, bool local)
{
}

void Surrogate::add(const vector<double> &x, double y)
{
  gp->add_pattern(&x[0], y);
}

void Surrogate::train_gp(libgp::GaussianProcess *gp_, bool log_fit)
{
  CG cg;
  cg.maximize(gp, 50, 0);
}

void Surrogate::train_gp_first()
{
  train_gp(NULL, false);
}

void Surrogate::train()
{
  train_gp(gp, false);
  }

double Surrogate::best_raw()
{
  return 0;
}

double Surrogate::error()
{
  return 0;
}

double Surrogate::var(double x[])
{
  return gp->var(x);
}

double Surrogate::mean(double x[])
{
  return gp->f(x);
}

Surrogate::~Surrogate()
{
  delete gp;
}
