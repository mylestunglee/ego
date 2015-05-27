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


EGO::EGO(int dim, Surrogate *s, vector<double> low, vector<double> up, double(*fit)(double x[]))
{
  dimension = dim;
  upper = up;
  lower = low;
  proper_fitness = fit;
  sg = s;
}

void EGO::run()
{
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing
    check_running_tasks();

    if(best_fitness <= max_fitness) {
      cout << "Found best at [";
      for(auto x : best_particle) {
        cout << x << ", ";
      }
      cout << "\b\b] with fitness [" << best_fitness << "]" << endl;
      break;
    }

    int lambda = num_lambda - running.size();
    lambda = min(lambda, max_iterations - num_iterations);

    if(is_new_result) {
      cout << "Iter: " << num_iterations << " / " << max_iterations;
      cout << ", RUNNING: " << running.size() << " Lambda: " << lambda;
      cout << " best " << best_fitness << endl;
      is_new_result = false;
    }

    vector<double> best_xs;
    if(use_brute_search) {
      best_xs = brute_search(10, lambda);
    } else { 
      best_xs = max_ei_par(lambda);
    }

    for(int l = 0; l < lambda; l++) {
      vector<double> y(dimension, 0.0);
      for(int j = 0; j < dimension; j++) {
        y[j] = best_xs[l * dimension + j];
      }

      if(not_running(&y[0]) && not_run(&y[0])) {
        evaluate(y);
      } else {
        cout << "Have run: ";
        for(int j = 0; j < dimension; j++) {
          cout << y[j] << " ";
        }
        y = local_brute_search(best_particle, 2, 1);
        evaluate(y);
      }
      cout << "Evaluating: ";
      for(int j = 0; j < dimension; j++) {
        cout << y[j] << " ";
      }
      cout << endl;
    }
  }
  check_running_tasks();
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
  num_iterations++;

  //Launch another thread to calculate
  thread (&EGO::worker_task, this, running.back(), num_iterations).detach();
  //worker_task(running.back(), num_iterations);
}

void EGO::worker_task(EGO::running_node node, int num)
{
  //Perform calculation
  vector<double> data_vec = node.data;
  double *data = &(data_vec[0]);
  double result = proper_fitness(data);

  running_mtx.lock();

  //Add results back to node keeping track
  vector<struct running_node>::iterator running_i = running.begin();
  while(running_i != running.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if(running_i->data[i] != node.data[i]) break;
    }
    if(i == dimension) {
      running_i->is_finished = true;
      running_i->fitness = result;
    }
    running_i++;
  }

  running_mtx.unlock();
}

void EGO::check_running_tasks()
{
  min_running = 1000000000;
  running_mtx.lock();

  vector<struct running_node>::iterator node = running.begin();

  while(node != running.end()) {
    if(node->is_finished) {
      //num_iterations++;
      for(int i = 0; i < dimension; i++) {
        cout << node->data[i] << " ";
      }
      cout << " evaluated to: " << node->fitness << endl;

      //Add it to our training set
      add_training(node->data, node->fitness);

      //Delete estimations
      mu_means.erase(mu_means.begin() + node->pos);
      mu_vars.erase(mu_vars.begin() + node->pos);
      for(int i = 0; i < running.size(); i++) {
        if(running[i].pos > node->pos) running[i].pos--;
      }

      //Delete node from running vector
      node = running.erase(node);
      is_new_result = true;
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

    if(not_running(y)) {
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
    best = brute_search(10, 1);
  } else {
    int x = dimension * lambda;
    vector<double> low(x, 0.0), up(x, 0.0);
    random_device rd;
    mt19937 gen(rd());

    for(int i = 0; i < x; i++) {
      low[i] = lower[i % dimension];
      up[i] = upper[i % dimension];
    }

    opt op(x, up, low, this, is_discrete);
    best = op.optimise(population_size);
  }

  return best;
}
vector<double> EGO::local_brute_search(vector<double> particle, int npts = 3, double radius = 1.0)
{
  double best = 1000000;
  double y_min = get_y_min();
  vector<double> best_point(dimension, 0);
  double points[dimension][npts + 1];
  bool has_result = false;
  for(int j = 0; j < dimension; j++) {
    if(is_discrete) {
      int step = floor((2 * radius) / npts);
      if(step == 0) step = 1;
      for(int i = 0; i <= npts; i++) {
        points[j][i] = particle[j] + (i - npts / 2) * step;
	points[j][i] = min(upper[i], points[j][i]);
	points[j][i] = max(lower[i], points[j][i]);
      }
    } else { 
      double step = (2 * radius) / npts;
      for(int i = 0; i <= npts; i++) {
        points[j][i] = particle[j] + (i - npts / 2) * step;
	points[j][i] = min(upper[i], points[j][i]);
	points[j][i] = max(lower[i], points[j][i]);
      }
    }
  }

  for(int i = 0; i < pow(npts, dimension); i++) {
    double x[dimension];
    for(int j = 0; j < dimension; j++) {
      x[j] = points[j][((int)(i / pow(npts, j))) % npts];
    }
    if(not_running(x) && not_run(x)) {
      double result = -ei(sg->_mean(x), sg->_var(x), y_min);
      if(result < best) {
        best = result;
        best_point.assign(x, x + dimension);
	has_result = true;
      }
    }
  }
  if(has_result) {
    return best_point;
  } else {
    return local_brute_search(particle, npts + 2, radius + 1);
  }
}

vector<double> EGO::brute_search(int npts=10, int lambda=1)
{
  double best = 1000000;
  double y_min = get_y_min();
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  double points[size][npts + 1];
  int num_steps = npts + 1;
  bool has_result = false;

  if(lambda == 1) {
    //Build our steps of possible values in each dimension
    for(int i = 0; i < dimension; i++) {
      if(is_discrete) {
        int step = floor((upper[i] - lower[i]) / npts);
        if(step == 0) step = 1;
        int j = 0;
        for(; j <= npts && (lower[i] + j * step) <= upper[i]; j++) {
          points[i][j] = floor(lower[i] + j * step);
        }
        num_steps = min(num_steps, j);
      } else {
        double step = (upper[i] - lower[i]) / npts;
        int j = 0;
        for(; j <= npts; j++) {
          points[i][j] = lower[i] + j * step;
        }
        num_steps = min(num_steps, j);
      }
    }

    //Loop through each possible combination of values
    for(int i = 0; i < pow(num_steps, dimension); i++) {
      double x[dimension];
      for(int j = 0; j < dimension; j++) {
        x[j] = points[j][((int)(i / pow(num_steps, j))) % num_steps];
      }
      if(not_running(x)) {
        double result = -ei(sg->_mean(x), sg->_var(x), y_min);
        if(result < best) {
          best = result;
          best_point.assign(x, x + dimension);
          has_result = true;
        }
      }
    }
  } else {
    //Build our steps of possible values in each dimension
    for(int i = 0; i < size; i++) {
      if(is_discrete) {
        int step = floor((upper[i % dimension] - lower[i % dimension]) / npts);
        if(step == 0) step = 1;
        int j = 0;
        for(; j <= npts && (lower[i % dimension] + j * step) <= upper[i % dimension]; j++) {
          points[i][j] = floor(lower[i % dimension] + j * step);
        }
        num_steps = min(num_steps, j);
      } else {
        double step = (upper[i % dimension] - lower[i % dimension]) / npts;
        int j = 0;
        for(; j <= npts; j++) {
          points[i][j] = lower[i % dimension] + j * step;
        }
        num_steps = min(num_steps, j);
      }
    }

    //Loop through each possible combination of values
    for(int i = 0; i < pow(num_steps, size); i++) {
      vector<double> x(size, 0.0);
      for(int j = 0; j < size; j++) {
        x[j] = points[j][((int) floor(i / pow(num_steps, j))) % num_steps];
      }
      double result = fitness(x);
      if(result < best) {
        best = result;
        best_point = x;
        has_result = true;
      }
    }
  }

  if(has_result) { 
    return best_point;
  } else {
    if(num_steps > npts || !is_discrete) {
      return brute_search(npts * 2, lambda);
    }
    cout << "Broken, can't brute search" << endl;
    exit(1);
  }
}

bool EGO::not_run(double x[])
{
  double eps = 0.0001;
  vector<vector<double>>::iterator train = training.begin();
  while(train != training.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if(abs((*train)[i] - x[i]) > eps) break;
    }
    if(i == dimension) return false;
    train++;
  }
  return true;
}

bool EGO::not_running(double x[])
{
  double eps = 0.0001;
  //vector<vector<double>>::iterator train = training.begin();
  //while(train != training.end()) {
  //  int i = 0;
  //  for(; i < dimension; i++) {
  //    if(abs((*train)[i] - x[i]) > eps) break;
  //  }
  //  if(i == dimension) return false;
  //  train++;
  //}
  vector<struct running_node>::iterator node = running.begin();
  while(node != running.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if(abs(node->data[i] - x[i]) > eps) break;
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
    double y_diff = y_min - y;
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
