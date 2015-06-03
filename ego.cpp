#include <iostream>
#include "surrogate.h"
#include "gp.h"
#include "ego.h"
#include "optimise.h"
#include "ihs.hpp"
#include <thread>
#include <Eigen/Dense>

using namespace std;

double gaussrand1();
double phi(double x);
double normal_pdf(double x);

const double PI = std::atan(1.0)*4;


EGO::EGO(int dim, shared_ptr<Surrogate> s, vector<double> low, vector<double> up, double(*fit)(double x[]))
{
  dimension = dim;
  upper = up;
  lower = low;
  proper_fitness = fit;
  sg = s;
}

void EGO::run()
{
  if(!suppress) cout << "Started, dim=" << dimension << ", lambda=" << num_lambda << endl;
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing
    check_running_tasks();

    if(best_fitness <= max_fitness) {
      if(!suppress) {
        cout << "Found best at [";
        for(auto x : best_particle) {
          cout << x << ", ";
        }
        cout << "\b\b] with fitness [" << best_fitness << "]" << endl;
      }
      while(running.size() > 0){
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
	check_running_tasks();
      }
      break;
    }

    int lambda = num_lambda - running.size();
    lambda = min(lambda, max_iterations - num_iterations);

    if(is_new_result && !suppress) {
      cout << "Iter: " << num_iterations << " / " << max_iterations;
      cout << ", RUNNING: " << running.size() << " Lambda: " << lambda;
      cout << " best " << best_fitness << endl;
      is_new_result = false;
    }

    if(lambda > 0) {
      vector<double> best_xs = max_ei_par(lambda);
      for(int l = 0; l < lambda; l++) {
        vector<double> y(dimension, 0.0);
        for(int j = 0; j < dimension; j++) {
          y[j] = best_xs[l * dimension + j];
        }

        if(not_running(&y[0]) && not_run(&y[0])) {
          evaluate(y);
        } else {
	  if(!suppress) {
            cout << "Have run: ";
            for(int j = 0; j < dimension; j++) {
              cout << y[j] << " ";
            }
	  }
          y = brute_search_local_swarm(best_particle, 2, 1, 1, true);
          evaluate(y);
        }
      }
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
  if(!suppress) {
    cout << "Evaluating: ";
    for(int j = 0; j < dimension; j++) {
      cout << x[j] << " ";
    }
    cout << endl;
  }
  double mean = sg->mean(data);
  mu_means.push_back(mean);
  mu_vars.push_back(sg->var(data));

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
  running_mtx.lock();

  vector<struct running_node>::iterator node = running.begin();
  while(node != running.end()) {
    if(node->is_finished) {
      if(!suppress) {
        for(int i = 0; i < dimension; i++) {
          cout << node->data[i] << " ";
        }
        cout << " evaluated to: " << node->fitness << endl;
      }

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
      lambda_means[i] = sg->mean(y);
      lambda_vars[i] = sg->var(y);
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
  //if(2*(num_iterations - max_points) > num_points * dimension) num_points = min(max_points, 2*num_points);
  if(lambda == 1) {
    vector<double> *x = brute_search_swarm(num_points, 1);
    if(x) {
      best = *x;
    } else {
      best = brute_search_local_swarm(best_particle, 4, 2.0, 1, true);
    }
  } else {
    if(use_brute_search) {
      vector<double> *ptr = NULL;
      if(swarm) {
        ptr = brute_search_swarm(num_points, lambda);
      } else {
        ptr = brute_search_loop(num_points, lambda, 0.01);
      }
      if(ptr) { 
        best = *ptr;
        delete ptr;
      } else {
        if(!suppress) cout << "Couldn't find new particles, searching in region of best" << endl;
        best = brute_search_local_swarm(best_particle, max(lambda, 4), max(lambda/2.0, 2.0), lambda, true);
        if(!suppress) {
          for(int i = 0; i < lambda * dimension; i++) {
            cout << best[i] << " ";
          }
          cout << " got around best" << endl;
        }
      }
    } else { 
      int size = dimension * lambda;
      vector<double> low(size, 0.0), up(size, 0.0), x(size, 0.0);
      random_device rd;
      mt19937 gen(rd());

      for(int i = 0; i < size; i++) {
        low[i] = lower[i % dimension];
        up[i] = upper[i % dimension];
        x[i] = best_particle[i % dimension];
      }

      opt op(size, up, low, this, is_discrete);
      best = op.swarm_optimise(x, lambda * dimension, lambda * population_size);

      if(!suppress) {
        cout << "[";
        for(int i = 0; i < lambda; i++) {
          for(int j = 0; j < dimension; j++) {
            cout << best[i*dimension + j] << " ";
          }
          cout << "\b; ";
        }
        cout << "\b\b] = best = "  << fitness(best) << endl;
      }
    }
  }

  return best;
}

vector<double> *EGO::brute_search_swarm(int npts, int lambda)
{
  double best = -0.5;
  int size = dimension * lambda;
  vector<double> *best_point = new vector<double>(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor((upper[i] - lower[i]) / npts);
      if(steps[i] == 0) steps[i] = 1.0;
    } else {
      steps[i] = (upper[i] - lower[i]) / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int j = 0; j < dimension; j++) {
        x[j] = lower[j] + floor((loop[0] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) * steps[j];
        if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
      }
      if(++loop[0] == pow(npts + 1, dimension)) more_viable = false;

      if(can_run) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < best) {
          best_point->assign(x.begin(), x.end());
          best = result;
          has_result = true;
        }
      }
    }

  } else {
    //lambda >= 2
    int num_loops = pow(npts + 1, dimension);
    for(int i = 0; i < lambda; i++) {
      best = -0.01;
      vector<double> point((i+1)*dimension, 0.0);
      bool found = false;

      for(int j = 0; j < i*dimension; j++) {
        point[j] = (*best_point)[j];
      }

      for(int j = 0; j < num_loops; j++) {
        bool can_run = true;
	//for(int k = 0; k < i; k++) {
	//  if(j == loop[k]) {
	//    j++;
	//    k = 0;
	//  }
	//}

        for(int k = 0; k < dimension; k++) {
          point[i * dimension + k] = lower[k] + floor((j % (int) pow(npts + 1, dimension - k)) / pow(npts + 1, dimension - k - 1)) * steps[k];
          if(point[i * dimension + k] > upper[k] || point[i * dimension + k] < lower[k]) can_run = false;
        }

        if(can_run) {
	  double result = 0.0;
	  if(i == 0) {
            double mean = sg->mean(&point[0]);
            double var = sg->var(&point[0]);
            result = -ei(mean, var, best_fitness);
	  } else {
            result = fitness(point);
	  }
          if(result < best) {
            for(int k = 0; k < dimension; k++) {
              (*best_point)[i*dimension + k] = point[i*dimension + k];
            }
            best = result;
	    loop[i] = j;
            if(i == lambda - 1) has_result = true;
	    found = true;
          }
        }
      }
      if(!found) {
	delete best_point;
        return NULL;
      }
    }
  }

  if(has_result) {
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << (*best_point)[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best << endl;
    }
    return best_point;
  } else {
    delete best_point;
    return NULL;
  }
}

void EGO::sample_plan(int F, int D)
{
  int* latin = ihs(dimension, F, D, D);
  for(int i = 0; i < F; i++) {
    vector<double> x(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      x[j] = lower[j] + latin[i*dimension+j];
    }
    while(running.size() == num_lambda) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      check_running_tasks();
    }
    evaluate(x);
  }
  while(training.size() < F) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check_running_tasks();
  }
  delete latin;
}

vector<double> EGO::brute_search_local(vector<double> particle, int npts, double radius, int lambda, bool has_to_run)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  int loops[size];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < size; i++) loops[i] = 0;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor(2 * radius / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = 2 * radius / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      bool can_run = true;
      vector<double> x(size, 0.0);

      for(int j = 0; j < size; j++) {
        x[j] = particle[j % dimension] + (loops[j] - npts / 2)  * steps[j % dimension];
	if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
      }

      for(int j = 0; j < size; j++) {
        if(++loops[j] == npts + 1) {
	  loops[j] = 0;
	  if(j == size - 1) {
	    more_viable = false;
	  }
	} else {
	  break;
	}
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
	double mean = sg->mean(&x[0]);
	double var = sg->var(&x[0]);
        double result = ei(mean, var, best_fitness);
        if(result < best) {
          best_point = x;
	  best = result;
	  has_result = true;
	}
      }
    }
  } else {
    while(more_viable) {
      bool can_run = true;
      vector<double> x(size, 0.0);

      for(int j = 0; j < size; j++) {
        x[j] = particle[j % dimension] + (loops[j] - npts / 2)  * steps[j % dimension];
	if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
      }

      for(int j = 0; j < size; j++) {
        if(++loops[j] == npts + 1) {
	  loops[j] = 0;
	  if(j == size - 1) {
	    more_viable = false;
	  }
	} else {
	  break;
	}
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double result = fitness(x);
        if(result < best) {
          best_point = x;
	  best = result;
	  has_result = true;
        }
      }
    }
  }
  if(has_result) {
    return best_point;
  } else if(has_to_run) {
    return brute_search_local(particle, npts, radius + 1, lambda, has_to_run);
  } else {
    return brute_search_local(particle, npts + 2, radius + 1, lambda, has_to_run);
  }
}

vector<double> EGO::brute_search_local_swarm(vector<double> particle, int npts, double radius, int lambda, bool has_to_run)
{
  double best = -0.01;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor(2 * radius / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = 2 * radius / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int j = 0; j < dimension; j++) {
        x[j] = particle[j] + (floor((loop[0] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) - npts/2) * steps[j];
        if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
      }

      if(++loop[0] == pow(npts + 1, dimension)) more_viable = false;

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < best) {
          best_point = x;
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    //lambda >= 2
    int num_loops = pow(npts + 1, dimension);
    for(int i = 0; i < lambda; i++) {
      best = -0.01;
      vector<double> point((i+1)*dimension, 0.0);
      bool found = false;

      for(int j = 0; j < i*dimension; j++) {
        point[j] = best_point[j];
      }

      for(int j = 0; j < num_loops; j++) {
        bool can_run = true;
	for(int k = 0; k < i; k++) {
	  if(j == loop[k]) {
	    j++;
	    k = 0;
	  }
	}

        for(int k = 0; k < dimension; k++) {
          point[i * dimension + k] = particle[i*dimension+k] + (floor((j % (int) pow(npts + 1, dimension - k)) / pow(npts + 1, dimension - k - 1)) - npts/2 )* steps[k];
          if(point[i * dimension + k] > upper[k] || point[i * dimension + k] < lower[k]) can_run = false;
        }

        if(can_run && (!has_to_run || (not_run(&point[i*dimension]) && not_running(&point[i*dimension])))) {
	  double result = 0.0;
	  if(i == 0) {
            double mean = sg->mean(&point[0]);
            double var = sg->var(&point[0]);
            result = -ei(mean, var, best_fitness);
	  } else {
            result = fitness(point);
	  }

          if(result < best) {
            best_point = point;
            best = result;
	    loop[i] = j;
            if(i == lambda - 1) has_result = true;
	    found = true;
          }
        }
      }
      if(!found) {
	break;
      }
    }
  }

  if(has_result) {
    return best_point;
  } else if(has_to_run) {
    return brute_search_local_swarm(particle, 2*(radius + 1), radius + 1, lambda, has_to_run);
  } else {
    return brute_search_local_swarm(particle, npts, radius + 1, lambda, has_to_run);
  }
}


vector<double> EGO::brute_search_local_loop(vector<double> particle, int npts, double radius, int lambda, bool has_to_run)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor(2 * radius / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = 2 * radius / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = particle[j] + (floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) - npts/2) * steps[j];
          if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[i] >= pow(npts + 1, dimension)) more_viable = false;
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < best) {
          best_point = x;
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    //lambda >= 2
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = particle[j] + (floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) - npts/2) * steps[j];
          if(x[i*dimension +j] > upper[j] || x[i*dimension+j] < lower[j]) can_run = false;
	}
	if(!can_run) break;
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[j] >= pow(npts + 1, dimension) + j - (lambda - 1)) more_viable = false;
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double result = fitness(x);
        if(result < best) {
          best_point = x;
	  best = result;
	  has_result = true;
        }
      }
    }
  }
  if(has_result) {
    return best_point;
  } else if(has_to_run) {
    return brute_search_local_loop(particle, 2*(radius + 1), radius + 1, lambda, has_to_run);
  } else {
    return brute_search_local_loop(particle, npts, radius + 1, lambda, has_to_run);
  }
}

vector<double>* EGO::brute_search_loop(int npts, int lambda, double min_ei)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> *best_point = new vector<double>(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor((upper[i] - lower[i]) / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = (upper[i] - lower[i]) / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = lower[j] + floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) * steps[j];
          if(x[i*dimension +j] > upper[j] || x[i*dimension+j] < lower[j]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[i] >= pow(npts + 1, dimension)) more_viable = false;
      }

      if(can_run) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < min(best, -min_ei)) {
          best_point->assign(x.begin(), x.end());
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    //lambda >= 2
    int num_loops = pow(npts + 1, dimension);
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = lower[j] + floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) * steps[j];
          if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (num_loops + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[j] >= num_loops) more_viable = false;
      }

      if(can_run) {
        double result = fitness(x);
        if(result < min(best, -min_ei)) {
          best_point->assign(x.begin(), x.end());
	  best = result;
	  has_result = true;
        }
      }
    }
  }

  if(has_result) {
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << (*best_point)[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best << endl;
    }
    return best_point;
  } else {
    delete best_point;
    return NULL;
  }
}

vector<double> EGO::brute_search(int npts=10, int lambda=1)
{
  double best = 1000000;
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
        double result = -ei(sg->mean(x), sg->var(x), best_fitness);
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
    int max_mus = mu_means.size();

    for (int k=0; k < n; k++) {
        double min = best_fitness;
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
	} else {
       	  Z = sqrt(-2 * log(U)) * cos(2 * PI * V);
	}

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
