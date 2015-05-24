#include "surrogate.h"
#include "gp.h"


using namespace std;
using namespace libgp;

Surrogate::Surrogate(int d, s_type type)
{
  dim = d;
  if(type == SEiso) {
    gp = new GaussianProcess(dim, "CovSEiso");
  } 
}

void Surrogate::set_params(double x, double y)
{
  Eigen::VectorXd params(gp->covf().get_param_dim());
  params << x, y;
  gp->covf().set_loghyper(params);
}

void Surrogate::add(vector<double> x, double y)
{
  double *data = &x[0];
  mtx.lock();
  gp->add_pattern(data, y);
  mtx.unlock();
}

double Surrogate::_var(double x[])
{
  mtx.lock();
  double result =  gp->var(x);
  mtx.unlock();
  return result;
}

double Surrogate::_mean(double x[]) 
{
  mtx.lock();
  double result =  gp->f(x);
  mtx.unlock();
  return result;
}
