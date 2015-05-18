#include <iostream>
#include "surrogate.h"
#include "gp.h"
#include "ego.h"
#include "optimise.h"
#include <Eigen/Dense>

using namespace std;

double gaussrand1();

const double PI = std::atan(1.0)*4;


EGO::EGO(int dim, vector<double> low, vector<double> up, double(*fit)(double x[]))
{
  dimension = dim;
  upper = up;
  lower = low;
  for(int i = 0; i < dimension; i++) {
    uniform_real_distribution<> dist(lower[i], upper[i]);
    _generator.push_back(dist);
  }
  proper_fitness = fit;
  sg = new Surrogate(dim);
}

void EGO::run()
{
  for(int iter = 0; iter < max_iterations; iter++) {

    int lambda = num_lambda - running.size();
    vector<double> best_xs = max_ei_par(lambda);

    for(int l = 0; l < lambda; l++) {
      double y [dimension];
      for(int j = 0; j < dimension; j++) {
        y[j] = best_xs[l * dimension + j];
      }
      fitness_function(y);
    }
  }
}

void EGO::fitness_function(double x[])
{
  double result = proper_fitness(x);
  sg->add(x, result);
}

double EGO::fitness(vector<double> x)
{
  int num = x.size() / dimension;
  double lambda_means[num];
  double lambda_vars[num];

  for(int i = 0; i < num; i++) {
    double y [dimension];

    for(int j = 0; j < dimension; j++) {
      y[j] = x[i * dimension + j];
    }

    lambda_means[i] = sg->_mean(y);
    lambda_vars[i] = sg->_var(y);
  }

  double result = ei_multi(lambda_vars, lambda_means, num, n_sims);
  return result / n_sims;
};

vector<double> EGO::max_ei_par(int lambda) 
{
  int dim = dimension;
  int x = dim * lambda;
  vector<double> low(x, 0.0), up(x, 0.0);
  random_device rd;
  mt19937 gen(rd());

  for(int i = 0; i < x; i++) {
    low[i] = lower[i % dim];
    up[i] = upper[i % dim];
  }

  opt op(x, up, low, this);
  vector<double> best = op.optimise(100);

  return best;
}

double EGO::ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n)
{
    double sum_ei=0.0, ei=0.0;
    double y_best = sg->y_best();
    int max_mus = mu_means.size();

    for (int k=0; k < n; k++) {
        double min = y_best;
        for(int i=0; i < max_mus; i++){
            double mius = gaussrand1()*mu_vars[i] + mu_means[i];
            if (mius < min)
                min = mius;
        }
        double min2=100000000.0;
        for(int j=0;j<max_lambdas;j++){
            double lambda = gaussrand1()*lambda_s2[j] + lambda_mean[j];
            if (lambda < min2)
                min2 = lambda;
        }
        
        ei = min - min2;
        if (ei < 0.0) {
          ei = 0.0;
	}
        sum_ei = ei + sum_ei;
    }
    return sum_ei;
}

//Use a method described by Abramowitz and Stegun:
double gaussrand1()
{
	static double U, V;
	static int phase = 0;
	double Z;

	if(phase == 0) {
		U = (rand() + 1.) / (RAND_MAX + 2.);
		V = rand() / (RAND_MAX + 1.);
		Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
	} else
		Z = sqrt(-2 * log(U)) * cos(2 * PI * V);

	phase = 1 - phase;

	return Z;
}
