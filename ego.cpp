#include <iostream>
#include "surrogate.h"
#include "gp.h"
#include "ego.h"
#include "optimise.h"
#include <thread>
#include <Eigen/Dense>

using namespace std;

double gaussrand1();
double phi(double x);
double normal_pdf(double x);

const double PI = std::atan(1.0)*4;


EGO::EGO(int dim, vector<double> low, vector<double> up, double(*fit)(double x[]))
{
  dimension = dim;
  upper = up;
  lower = low;
  proper_fitness = fit;
  sg = new Surrogate(dim);
}

void EGO::run()
{
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing
    check_running_tasks();

    if(best_fitness <= max_fitness) {
      cout << "Found best" << endl;
      break;
    }

    cout << "Iter: " << num_iterations << " / " << max_iterations << endl;

    int lambda = num_lambda - running.size();
    cout << "RUNNING: " << running.size() << " Lambda: " << lambda << endl;
    vector<double> best_xs = max_ei_par(lambda);

    for(int l = 0; l < lambda; l++) {
      vector<double> y(dimension, 0.0);
      cout << "Evaluating: ";
      for(int j = 0; j < dimension; j++) {
        y[j] = best_xs[l * dimension + j];
	cout << y[j] << " ";
      }
      cout << endl;
      evaluate(y);
      
    }

  }
}

vector<double> EGO::best_result()
{
  return best_particle;
}

void EGO::evaluate(vector<double> x)
{
  double *data = &x[0];
  double mean = sg->_mean(data);
  min_running = min(min_running, mean);
  mu_means.push_back(mean);
  mu_vars.push_back(sg->_var(data));

  //Add to running set
  struct running_node run; 
  run.fitness = mean;
  run.is_finished = false;
  run.data = x;
  run.pos = mu_means.size() - 1;
  running.push_back(run);

  //Launch another thread to calculate
  thread (&EGO::worker_task, this, &running.back()).detach();
}

void EGO::worker_task(EGO::running_node *node)
{
  //Perform calculation
  double *data = &(node->data[0]);
  double result = proper_fitness(data);

  running_mtx.lock();

  //Add results back to node keeping track
  node->is_finished = true;
  node->fitness = result;

  running_mtx.unlock();
}

void EGO::check_running_tasks()
{
  min_running = 1000000000;
  running_mtx.lock();

  vector<struct running_node>::iterator node = running.begin();

  while(node != running.end()) {
    if(node->is_finished) {
      num_iterations++;
      //Add it to our training set
      add_training(node->data, node->fitness);

      //Delete estimations
      mu_means.erase(mu_means.begin() + node->pos);
      mu_vars.erase(mu_vars.begin() + node->pos);

      //Delete node from running vector
      node = running.erase(node);
    } else {
      //Recalculate minimum estimate
      min_running = min(min_running, node->fitness);
      node++;
    }
  }
  
  running_mtx.unlock();
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

    if(not_run(y)) {
      lambda_means[i] = sg->_mean(y);
      lambda_vars[i] = sg->_var(y);
    } else {
      lambda_means[i] = 100000000000;
      lambda_vars[i] = 0;
    }
  }

  double result = -1 * ei_multi(lambda_vars, lambda_means, num, n_sims);
  return result / n_sims;
};

void EGO::add_training(vector<double> x, double y)
{
  training.push_back(x);
  training_fitness.push_back(y);
  sg->add(x, y);
  if(y < best_fitness) {
    best_fitness = y;
    best_particle = x;
  }
}

vector<double> EGO::max_ei_par(int lambda) 
{
  vector<double> best;
  if(lambda <= 1) {
    best = brute_search(10);
  } else {
    int x = dimension * lambda;
    vector<double> low(x, 0.0), up(x, 0.0);
    random_device rd;
    mt19937 gen(rd());

    for(int i = 0; i < x; i++) {
      low[i] = lower[i % dimension];
      up[i] = upper[i % dimension];
    }

    opt op(x, up, low, this);
    best = op.optimise(100);
  }

  return best;
}

vector<double> EGO::brute_search(int npts=10)
{
  double best = -1000000;
  double y_min = get_y_min();
  vector<double> best_point(dimension, 0);
  double points[dimension][npts];

  //Build our steps of possible values in each dimension
  for(int i = 0; i < dimension; i++) {
    double step = (upper[i] - lower[i]) / npts;
    for(int j= 0; j < npts; j++) {
      points[i][j] = lower[i] + j * step;
    }
  }

  //Loop through each possible combination of values
  for(int i = 0; i < pow(npts, dimension); i++) {
    double x[dimension];
    for(int j = 0; j < dimension; j++) {
      x[j] = points[j][((int)(i / pow(npts, j))) % npts];
    }
    if(not_run(x)) {
      double result = ei(sg->_mean(x), sg->_var(x), y_min);
      if(result > best) {
        best = result;
        best_point.assign(x, x + dimension);
      }
    }
  }
  return best_point;
}

bool EGO::not_run(double x[])
{
  vector<vector<double>>::iterator train = training.begin();
  while(train != training.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if((*train)[i] != x[i]) break;
    }
    if(i == dimension) return false;
    train++;
  }
  vector<struct running_node>::iterator node = running.begin();
  while(node != running.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if(node->data[i] != x[i]) break;
    }
    if(i == dimension) return false;
    node++;
  }
  //Not in training or running set, so hasn't been run
  return true;
  
}

double EGO::get_y_min()
{
  return min(best_fitness, min_running);
}

double EGO::ei(double y, double S2, double y_min) 
{
  if(S2 <= 0.0) {
    return 0.0;
  } else {
    double s = sqrt(S2);
    double y_diff = y - y_min;
    double y_diff_s = y_diff / s;
    return y_diff * phi(y_diff_s) + s * normal_pdf(y_diff_s);
  }
}

double EGO::ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n)
{
    double sum_ei=0.0, e_i=0.0;
    double y_best = get_y_min();
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
        
        e_i = min - min2;
        if (e_i < 0.0) {
          e_i = 0.0;
	}
        sum_ei = e_i + sum_ei;
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

 
//Code for CDF of normal distribution
double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
 
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);
 
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
 
    return 0.5*(1.0 + sign*y);
}

double normal_pdf(double x)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;

    return inv_sqrt_2pi * exp(-0.5 * x * x);
}
