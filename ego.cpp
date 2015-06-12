#include <iostream>
#include "surrogate.h"
#include "gp.h"
#include "ego.h"
#include "optimise.h"
#include "ihs.hpp"

#define _GLIBCXX_USE_NANOSLEEP //Need to send thread to sleep on old GCC

#include <thread>
#include <chrono>
#include <Eigen/Dense>


using namespace std;

double gaussrand1();
double phi(double x);
double normal_pdf(double x);

const double PI = std::atan(1.0)*4;

EGO::~EGO()
{
  delete sg;
  if(pModule) {
    // Clean up
    Py_DECREF(pModule);

    // Finish the Python Interpreter
    Py_Finalize();
    delete sg_cost;
  }
}

void EGO::run_quad()
{
  if(!suppress) cout << "Started, dim=" << dimension << ", lambda=" << num_lambda << endl;
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing

    auto t1 = std::chrono::high_resolution_clock::now();
    sg->train();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    update_running(t3);

    if(at_optimum || best_fitness <= max_fitness) {
      if(!suppress) {
        cout << "Found best at [";
        for(int i = 0; i < dimension; i++) {
          cout << best_particle[i] << ", ";
        }
        cout << "\b\b] with fitness [" << best_fitness << "]" << endl;
      }
      while(running.size() > 0){
	update_running(10000000);
      }
      break;
    }

    if(is_new_result && !suppress) {
      cout << "Iter: " << num_iterations << " / " << max_iterations;
      cout << ", RUNNING: " << running.size() << " Lambda: " << lambda;
      cout << " best " << best_fitness << endl;
      is_new_result = false;
    }

    if(lambda > 0) {
      t1 = std::chrono::high_resolution_clock::now();
      vector<double> best_xs = max_ei_par(lambda);
      t2 = std::chrono::high_resolution_clock::now();
      t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
      update_running(t3);

      t1 = std::chrono::high_resolution_clock::now();
      t2 = std::chrono::high_resolution_clock::now();
      for(int l = 0; l < lambda; l++) {
        vector<double> y(dimension, 0.0);
        for(int j = 0; j < dimension; j++) {
          y[j] = best_xs[l * dimension + j];
        }

        if(not_running(&y[0]) && not_run(&y[0])) {
          python_eval(y);
        } else {
	  if(!suppress) {
            cout << "Have run: ";
            for(int j = 0; j < dimension; j++) {
              cout << y[j] << " ";
            }
	  }
          y = brute_search_local_swarm(best_particle, 1, 1, true);
          python_eval(y);
        }
      }
      t2 = std::chrono::high_resolution_clock::now();
      t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
      update_running(t3);
    }
    update_running();
  }
}

void EGO::python_eval(const vector<double> &x, bool add)
{
  double *data = (double *) &x[0];
  if(!suppress) {
    cout << "Evaluating: ";
    for(int j = 0; j < dimension; j++) {
      cout << x[j] << " ";
    }
    cout << endl;
  }

  double mean, var;
  if(sg->is_trained){
    pair<double, double> p = sg->predict(data);
    mean = p.first;
    var = p.second;
  } else {
    mean = sg->mean(data);
    var = sg->var(data);
  }
  mu_means.push_back(mean);
  mu_vars.push_back(var);

  struct running_node run; 
  PyObject *args, *index, *args2, *fitness_result, *results, *temp, *index_0;
  PyObject *args = PyTuple_New(x.size());
  //Set up arguments for call to fitness function
  for(int i = 0; i < x.size(); i++) {
    pValue = PyFloat_FromLong(x[i]);
    if(!pValue) {
      cout << "Broken python code in eval, exiting" << endl;
      cout << "Broke on adding " << x[i] << " to arg list" << endl;
      Py_DECREF(pValue);
      Py_DECREF(args);
      exit(-1);
    }
    index = PyInt_FromLong(i);
    PyTuple_SetItem(args, index, value);
    Py_DECREF(index);
    Py_DECREF(pValue);
  }

  //SO LONG URGH
  args2 = PyTuple_New(2);
  index = PyInt_FromLong(0);
  PyTuple_SetItem(args, index, args);
  Py_DECREF(index);
  index = PyInt_FromLong(1);
  if(pState) {
    PyTuple_SetItem(args, index, pState);
  } else {
    PyTuple_SetItem(args, index, Py_None);
  }
  Py_DECREF(index);

  //Finally call python code
  fitness_result = PyObject_CallObject(pFunc, args);

  if(args) Py_DECREF(args);
  if(pState) Py_DECREF(pState);
  //Check dem errors
  int size = PyObject_Size(fitness_result);
  if(size < 2) {
    cout << "Broken python code in eval, exiting" << endl;
    cout << "Broke on reading returned fitness values" << endl;
    py_exit();
  }
  index = PyInt_FromLong(0);
  results = PyObject_GetItem(fitness_result, index);
  Py_DECREF(index);

  //Set all the python stuff back to C++ types
  index = PyInt_FromLong(1);
  if(pState) Py_DECREF(pState);
  //Set our state object
  pState = PyObject_GetItem(fitness_result, index);
  Py_DECREF(index);

  //Grab all those juicy results - now in longform
  index_0 = PyInt_FromLong(0);
  pValue = PyObject_GetItem(results, index);
  temp = PyObject_GetItem(pValue, index_0);
  run.fitness = PyFloat_AsDouble(temp);
  Py_DECREF(pValue);
  Py_DECREF(temp);

  index = PyInt_FromLong(1);
  pValue = PyObject_GetItem(results, index);
  temp = PyObject_GetItem(pValue, index_0);
  run.label = PyInt_AsLong(temp);
  Py_DECREF(pValue);
  Py_DECREF(temp);
  Py_DECREF(index);

  index = PyInt_FromLong(2);
  pValue = PyObject_GetItem(results, index);
  temp = PyObject_GetItem(pValue, index_0);
  run.addReturn = PyInt_AsLong(temp);
  Py_DECREF(pValue);
  Py_DECREF(temp);
  Py_DECREF(index);

  index = PyInt_FromLong(3);
  pValue = PyObject_GetItem(results, index);
  temp = PyObject_GetItem(pValue, index_0);
  run.cost = PyInt_AsLong(pValue);
  Py_DECREF(pValue);
  Py_DECREF(temp);
  Py_DECREF(index);
  Py_DECREF(index0);
  Py_DECREF(results);
  Py_DECREF(fitness_result);
  //FINALLY CLEAN

  if(add) {
    num_iterations++;
    string str(x.begin(), x.end());
    cout << str << " fitness:" << run.fitness << " code:" << run.label << endl;
    sg->add(x, run.fitness, run.label + 1, run.addReturn);
    sg_cost->add(x, run.cost);
  } else {
    //Add to running set
    run.is_finished = false;
    run.data = x;
    run.pos = mu_means.size() - 1;
    running.push_back(run);
    num_iterations++;
  }
}

void update_running(chrono::duration<int> time)
{
  if(time = -1) {
    time = 1000000000000000;
    for(vector<struct running_node>::iterator node = running.begin(); node !=
    running.end();) time = min(time, node->cost);
  }
  for(vector<struct running_node>::iterator node = running.begin(); node != running.end();) {
    node->cost -= time;
    if(node->cost <= 0) {
      python_eval(node->data, true);
    }
  }
}

EGO::EGO(int dim, Surrogate *s, vector<double> low, vector<double> up, string python_file_name)
{
  dimension = dim;
  upper = up;
  lower = low;
  proper_fitness = fit;
  sg = s;
  n_sims = 50;
  max_iterations = 500;
  num_iterations = 0;
  num_lambda = 3;
  lambda = num_lambda;
  population_size = 100;
  num_points = 10;
  max_points = 10;
  pso_gen = 1;
  iter = 0;
  best_fitness = 100000000;
  max_fitness = 0;
  is_discrete = false;
  is_new_result = false;
  use_brute_search = true;
  suppress = false;
  at_optimum = false;

  sg_cost = new Surrogate(dim, SEiso);

  // Initialize the Python Interpreter
  Py_Initialize();

  // Build the name object
  pName = PyString_FromString(python_file_name);

  // Load the module object
  pModule = PyImport_Import(pName);

  // pDict is a borrowed reference 
  //pDict = PyModule_GetDict(pModule);

  pFunc = PyModule_GetAttrString(pModule, (char*)"fitnessFunc");

  Py_DECREF(pName);
}

EGO::EGO(int dim, Surrogate *s, vector<double> low, vector<double> up, double(*fit)(double x[]))
{
  dimension = dim;
  upper = up;
  lower = low;
  proper_fitness = fit;
  sg = s;
  n_sims = 50;
  max_iterations = 500;
  num_iterations = 0;
  num_lambda = 3;
  population_size = 100;
  num_points = 10;
  max_points = 10;
  pso_gen = 1;
  iter = 0;
  best_fitness = 100000000;
  max_fitness = 0;
  is_discrete = false;
  is_new_result = false;
  use_brute_search = true;
  suppress = false;
  at_optimum = false;
  pName = NULL;
}

void EGO::run()
{
  if(!suppress) cout << "Started, dim=" << dimension << ", lambda=" << num_lambda << endl;
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing
    check_running_tasks();
    sg->train();

    if(at_optimum || best_fitness <= max_fitness) {
      if(!suppress) {
        cout << "Found best at [";
        for(int i = 0; i < dimension; i++) {
          cout << best_particle[i] << ", ";
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
          y = brute_search_local_swarm(best_particle, 1, 1, true);
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

void EGO::evaluate(const vector<double> &x)
{
  double *data = (double *) &x[0];
  if(!suppress) {
    cout << "Evaluating: ";
    for(int j = 0; j < dimension; j++) {
      cout << x[j] << " ";
    }
    cout << endl;
  }

  double mean, var;
  if(sg->is_trained){
    pair<double, double> p = sg->predict(data);
    mean = p.first;
    var = p.second;
  } else {
    mean = sg->mean(data);
    var = sg->var(data);
  }
  mu_means.push_back(mean);
  mu_vars.push_back(var);

  //Add to running set
  struct running_node run; 
  run.fitness = mean;
  run.is_finished = false;
  run.data = x;
  run.pos = mu_means.size() - 1;
  running.push_back(run);
  num_iterations++;

  //Launch another thread to calculate
  std::thread (&EGO::worker_task, this, x).detach();
  
  //worker_task(x);
}

void EGO::worker_task(vector<double> node)
{
  //Perform calculation
  double result = proper_fitness(&node[0]);

  running_mtx.lock();

  //Add results back to node keeping track
  vector<struct running_node>::iterator running_i = running.begin();
  while(running_i != running.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if(running_i->data[i] != node[i]) break;
    }
    if(i == dimension) {
      running_i->is_finished = true;
      running_i->fitness = result;
      running_i->label = 1; //(int) (result < 100 && result > -100);
      break;
    }
    running_i++;
  }

  //Uses C++11 features and can't compile on old gcc
  //for(auto &run : running) {
  //  int i = 0;
  //  for(; i < dimension; i++) {
  //    if(run.data[i] != node[i]) break;
  //  }
  //  if(i == dimension) {
  //    run.is_finished = true;
  //    run.fitness = result;
  //    break;
  //  }
  //}

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
        cout << " evaluated to: " << node->fitness << " with label: " << node->label << endl;
      }

      //Add it to our training set
      add_training(node->data, node->fitness, node->label);

      //Delete estimations
      mu_means.erase(mu_means.begin() + node->pos);
      mu_vars.erase(mu_vars.begin() + node->pos);
      for(size_t i = 0; i < running.size(); i++) {
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

double EGO::fitness(const vector<double> &x)
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
      pair<double, double> p = sg->predict(y);
      lambda_means[i] = p.first;
      lambda_vars[i] = p.second;
    } else {
      lambda_means[i] = 100000000000;
      lambda_vars[i] = 0;
    }
  }

  double result = -1. * ei_multi(lambda_vars, lambda_means, num, n_sims);
  return result / n_sims;
}

void EGO::add_training(const vector<double> &x, double y, int label)
{
  training.push_back(x);
  training_fitness.push_back(y);
  sg->add(x, y, label);
  if(label == 1 && y < best_fitness) {
    best_fitness = y;
    best_particle = x;
  }
}

vector<double> EGO::max_ei_par(int lambda) 
{
  vector<double> best;
  //if(lambda == 1) {
  //  vector<double> *x = brute_search_swarm(num_points, 1);
  //  if(x) {
  //    best = *x;
  //  } else {
  //    if(!suppress) cout << "Locally ";
  //    best = brute_search_local_swarm(best_particle, 1, 1, true);
  //  }
  //} else {
    if(use_brute_search) {
      vector<double> *ptr = brute_search_swarm(num_points, lambda);
      if(ptr) { 
        best = *ptr;
        delete ptr;
      } else {
        if(!suppress) cout << "Couldn't find new particles, searching in region of best" << endl;
        best = brute_search_local_swarm(best_particle, lambda, lambda, true);
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

      opt *op = new opt(size, up, low, this, is_discrete);
      //auto t1 = std::chrono::high_resolution_clock::now();
      best = op->swarm_optimise(x, pso_gen * size, population_size, 200);
      //auto t2 = std::chrono::high_resolution_clock::now();
      //auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
      //cout << "PSO max_ei_par lambda=" << lambda << " took " << t3  << endl;
      double best_fitness = op->best_part->best_fitness;
      //cout << "Optimum fitness= " << op->best_part->best_fitness << " gen= " << op->last_gen << "/" << min(pso_gen*size, op->last_gen+400) << endl;

      if(!suppress) {
        cout << "[";
        for(int i = 0; i < lambda; i++) {
          for(int j = 0; j < dimension; j++) {
            cout << best[i*dimension + j] << " ";
          }
          cout << "\b; ";
        }
        cout << "\b\b] = best = "  << best_fitness << endl;
      }
    delete op;
    }
  //}

  iter++;
  return best;
}

void EGO::sample_plan(size_t F, int D)
{
  int* latin = ihs(dimension, F, D, D);
  if(!latin) {
    cout << "Sample plan broke horribly, exiting" << endl;
    exit(-1);
  }
  double frac = (upper[0] - lower[0]) / (F-1);
  for(size_t i = 0; i < F; i++) {
    vector<double> x(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      x[j] = lower[j] + frac*(latin[i*dimension+j]-1);
    }
    while(running.size() == num_lambda) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      check_running_tasks();
    }
    evaluate(x);
  }
  while(training.size() < F) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check_running_tasks();
  }
  delete latin;
  sg->train();
}

vector<double> *EGO::brute_search_swarm(int npts, int lambda)
{
  double best = -0.0;
  int size = dimension * lambda;
  vector<double> *best_point = new vector<double>(size, 0);
  unsigned long long npts_plus[dimension + 1];
  //unsigned long long loop[lambda];
  double steps[dimension];
  bool has_result = false;
  //for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor((upper[i] - lower[i]) / npts);
      if(steps[i] == 0) steps[i] = 1.0;
    } else {
      steps[i] = (upper[i] - lower[i]) / npts;
    }
    npts_plus[i] = (int) pow(npts + 1, dimension - i);
  }
  npts_plus[dimension] = 1;

  if(lambda == 1) {
    vector<double> x(size, 0.0);
    auto t1 = std::chrono::high_resolution_clock::now();
    for(unsigned long long i = 0; i < npts_plus[0]; i++) {
      bool can_run = true;
      for(int j = 0; j < dimension; j++) {
        x[j] = lower[j] + floor((i % npts_plus[j]) / npts_plus[j+1]) * steps[j];
        if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
      }

      if(can_run) {
        pair<double, double> p = sg->predict(&x[0]);
        double result = -ei(p.first, p.second, best_fitness);
        if(result < best) {
          best_point->assign(x.begin(), x.end());
          best = result;
          has_result = true;
        }
      }
    }
   auto t2 = std::chrono::high_resolution_clock::now();
   auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
   cout << "Lambda =  1 time = " << t3 << endl;
  } else {
    double lambda_means[lambda];
    double lambda_vars[lambda];
    pair<double, double> best_mean_var;
    for(int i = 0; i < lambda; i++) {
      auto t1 = std::chrono::high_resolution_clock::now();
      best = -0.001;
      vector<double> point((i+1)*dimension, 0.0);
      bool found = false;

      for(int j = 0; j < i*dimension; j++) {
        point[j] = (*best_point)[j];
      }

      for(unsigned long long j = 0; j < npts_plus[0]; j++) {
        bool can_run = true;
	//for(int k = 0; k < i; k++) {
	//  if(j == loop[k]) {
	//    j++;
	//    k = 0;
	//  }
	//}

        for(int k = 0; k < dimension; k++) {
          point[i * dimension + k] = lower[k] + floor((j % npts_plus[k]) / npts_plus[k+1]) * steps[k];
          if(point[i * dimension + k] > upper[k] || point[i * dimension + k] < lower[k]) can_run = false;
        }

        if(can_run) {
	  double result = 0.0;
	  pair<double, double> p = sg->predict(&point[i*dimension]);
	  if(i == 0) {
            result = -ei(p.first, p.second, best_fitness);
	  } else {
	    lambda_means[i] = p.first;
	    lambda_vars[i] = p.second;
            result = -1. * ei_multi(lambda_vars, lambda_means, i + 1, n_sims);
	  }
          if(result < best) {
            for(int k = 0; k < dimension; k++) {
              (*best_point)[i*dimension + k] = point[i*dimension + k];
            }
	    best_mean_var = p;
            best = result;
	    //loop[i] = j;
            if(i == lambda - 1) has_result = true;
	    found = true;
          }
        }
      }
      lambda_means[i] = best_mean_var.first;
      lambda_vars[i] = best_mean_var.second;
      auto t2 = std::chrono::high_resolution_clock::now();
      auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
      cout << "Lambda =  " << (i+1) << "/" << lambda << " time = " << t3 << endl;
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

vector<double> EGO::brute_search_local_swarm(const vector<double> &particle, double radius, int lambda, bool has_to_run)
{
  double best = -0.0;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  unsigned long long loop[lambda];
  double steps[dimension];
  unsigned long long npts_plus[dimension + 1];
  bool has_result = false;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = 1;
      npts_plus[i] = pow(2*radius + 1, dimension - i);
    } else {
      steps[i] = 2 * radius / max_points;
      npts_plus[i] = pow(max_points + 1, dimension - i);
    }
  }
  npts_plus[dimension] = 1;

  if(lambda == 1) {
    for(unsigned long long i = 0; i < npts_plus[0]; i++) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int j = 0; j < dimension; j++) {
        x[j] = particle[j] + floor(((i % npts_plus[j]) / npts_plus[j+1]) - radius) * steps[j];
        if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
	pair<double, double> p = sg->predict(&x[0]);
        double result = -ei(p.first, p.second, best_fitness);
        if(result < best) {
          best_point = x;
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    for(int i = 0; i < lambda; i++) {
      best = -0.0;
      vector<double> point((i+1)*dimension, 0.0);
      bool found = false;

      for(int j = 0; j < i*dimension; j++) {
        point[j] = best_point[j];
      }

      for(unsigned long long j = 0; j < npts_plus[0]; j++) {
        bool can_run = true;
	for(int k = 0; k < i; k++) {
	  if(j == loop[k]) {
	    j++;
	    k = 0;
	  }
	}

        for(int k = 0; k < dimension; k++) {
          point[i * dimension + k] = particle[k] + floor(((j % npts_plus[k]) / npts_plus[k+1]) - radius) * steps[k];
          if(point[i * dimension + k] > upper[k] || point[i * dimension + k] < lower[k]) can_run = false;
        }

        if(can_run && (!has_to_run || (not_run(&point[i*dimension]) && not_running(&point[i*dimension])))) {
	  double result = 0.0;
	  if(i == 0) {
	    pair<double, double> p = sg->predict(&point[0]);
            result = -ei(p.first, p.second, best_fitness);
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
  } else {
    if(is_discrete) {
      if(radius / lambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
        at_optimum = true;
        return best_point;
      } else {
        //cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, lambda, has_to_run);
      }
    } else {
      if(lambda == 1 && radius < 3) {
	max_points *= 2;
        return brute_search_local_swarm(particle, radius + 1, lambda, has_to_run);
      } else if(radius / lambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
        at_optimum = true;
        return best_point;
      } else {
        //cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, lambda, has_to_run);
      }
    }
  }
}

bool EGO::not_run(double x[])
{
  static double eps = std::numeric_limits<double>::epsilon();
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
  static double eps = std::numeric_limits<double>::epsilon();
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
