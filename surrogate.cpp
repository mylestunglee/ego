#include "surrogate.h"
#include "gp.h"


using namespace std;
using namespace libgp;

Surrogate::Surrogate(int d)
{
  dim = d;
  gp = new GaussianProcess(dim, "CovSEiso");
  Eigen::VectorXd params(gp->covf().get_param_dim());
  params <<  0.1, 2.0;
  gp->covf().set_loghyper(params);
  _y_best = 100000;
}

void Surrogate::add(double x[], double y)
{
  gp->add_pattern(x, y);
  vector<double> data(x, x + dim);
  pair<vector<double>, double> trained(data, y);
  training.push_back(trained);

  if(y < _y_best) {
    _y_best = y;
  }
}

double Surrogate::_var(double x[])
{
  return gp->var(x);
}

double Surrogate::_mean(double x[]) 
{
  return gp->f(x);
}

double Surrogate::y_best()
{
  return _y_best;
}

