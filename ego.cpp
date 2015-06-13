#include "ego.h"
#include "optimise.h"
#include "ihs.hpp"

#define _GLIBCXX_USE_NANOSLEEP //Need to send thread to sleep on old GCC

#include <thread>
#include <chrono>


using namespace std;

double gaussrand1();
double phi(double x);
double normal_pdf(double x);

const double PI = std::atan(1.0)*4;

EGO::~EGO()
{
  delete sg;
  if(pFunc) {
    // Clean up
    Py_DECREF(pModule);
    Py_DECREF(pFunc);

    // Finish the Python Interpreter
    Py_Finalize();
    delete sg_cost;
  }
}

void EGO::run_quad()
{
  if(!suppress) cout << "Started, dim=" << dimension << ", lambda=" <<
  num_lambda << " max_f=" << max_fitness << endl;
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing

    auto t1 = std::chrono::high_resolution_clock::now();
    sg->train();
    sg_cost->train();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
    update_running(t3);

    if(at_optimum || best_fitness <= max_fitness) {
      if(!suppress) {
        cout << "Found best at [";
        for(int i = 0; i < dimension; i++) {
          cout << best_particle[i] << ", ";
        }
        cout << "\b\b] with fitness [" << best_fitness << "]" << endl;
      }
      exit(0);
    }

    if(is_new_result && !suppress) {
      cout << "Iter: " << num_iterations << " / " << max_iterations;
      cout << ", RUNNING: " << running.size() << " Lambda: " << lambda;
      cout << " best " << best_fitness << endl;
      is_new_result = false;
    }

    if(lambda > 0) {
      int temp_lambda = lambda;
      t1 = std::chrono::high_resolution_clock::now();
      vector<double> best_xs = max_ei_par(temp_lambda);
      t2 = std::chrono::high_resolution_clock::now();
      t3 = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
      update_running(t3);

      t1 = std::chrono::high_resolution_clock::now();
      t2 = std::chrono::high_resolution_clock::now();
      for(int l = 0; l < temp_lambda; l++) {
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
      t3 = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
      update_running(t3);
    }
    update_running();
    if(running.size() >= num_lambda) {
      cout << "Couldn't update running, exiting" << endl;
      exit(-1);
    }
  }
}

void EGO::python_eval(const vector<double> &x, bool add)
{
  if(!add) {
    double *data = (double *) &x[0];
    if(!suppress) {
      cout << "Evaluating: ";
      for(int j = 0; j < dimension; j++) {
        cout << x[j] << " ";
      }
      cout << endl;
    }

    pair<double, double> p = sg->predict(data);
    mu_means.push_back(p.first);
    mu_vars.push_back(p.second);
  }

  struct running_node run; 
  PyObject *args, *index, *args2, *fitness_result, *results, *temp, *index_0;
  args = PyList_New(x.size());
  //Set up arguments for call to fitness function
  for(unsigned int i = 0; i < x.size(); i++) {
    pValue = PyInt_FromLong((long) round(x[i]));
    if(!pValue) {
      cout << "Broken python code in eval, exiting" << endl;
      cout << "Broke on adding " << x[i] << " to arg list" << endl;
      PyErr_Print();
      Py_DECREF(pValue);
      Py_DECREF(args);
      exit(-1);
    }
    PyList_SetItem(args, i, pValue);
    Py_DECREF(pValue);
  }

  //SO LONG URGH
  if(pState) {
    args2 = Py_BuildValue("(O,O)", args, pState);
  } else {
    cout << "Sending Py_None" << endl;
    args2 = Py_BuildValue("(O,O)", args, Py_None);
  }
  if(!args2) {
    cout << "Broken python code in eval, exiting" << endl;
    PyErr_Print();
    cout << "Couldn't create argument list" << endl;
    exit(-1);
  }

  //Finally call python code
  fitness_result = PyObject_CallObject(pFunc, args2);

  if(!fitness_result) {
    cout << "Broken python code in eval, exiting" << endl;
    PyErr_Print();
    cout << "No fitness values were returned" << endl;
    exit(-1);
  }

  if(args)  {
    Py_DECREF(args);
  }
  if(args2)  {
    Py_DECREF(args);
  }
  if(pState) {
    Py_DECREF(pState);
  }
    
  //Check dem errors
  int size = PyTuple_Size(fitness_result);
  if(size < 2) {
    cout << "Broken python code in eval, exiting" << endl;
    cout << "Broke on reading returned fitness values, size is " << size << endl;
    PyErr_Print();
    exit(-1);
  }
  index = PyInt_FromLong(0);
  results = PyObject_GetItem(fitness_result, index);
  Py_DECREF(index);

  if(pState) {
    Py_DECREF(pState);
  }

  //Set all the python stuff back to C++ types
  index = PyInt_FromLong(1);
  //Set our state object
  pState = PyObject_GetItem(fitness_result, index);
  if(!pState) {
    cout << "Broken python code in eval, exiting" << endl;
    PyErr_Print();
    cout << "No state returned" << endl;
    exit(-1);
  }
  Py_DECREF(index);

  //Grab all those juicy results - now in longform
  index_0 = PyInt_FromLong(0);
  pValue = PyObject_GetItem(results, index_0);
  if(!pValue) {
    cout << "Broken python code in eval, exiting" << endl;
    PyErr_Print();
    cout << "No fitness returned" << endl;
    exit(-1);
  }
  temp = PyObject_GetItem(pValue, index_0);
  run.fitness = PyFloat_AsDouble(temp);

  Py_DECREF(pValue);
  Py_DECREF(temp);

  index = PyInt_FromLong(1);
  pValue = PyObject_GetItem(results, index);
  if(!pValue) {
    cout << "Broken python code in eval, exiting" << endl;
    PyErr_Print();
    cout << "No label returned" << endl;
    exit(-1);
  }
  temp = PyObject_GetItem(pValue, index_0);
  run.label = PyInt_AsLong(temp);

  Py_DECREF(pValue);
  Py_DECREF(temp);
  Py_DECREF(index);

  index = PyInt_FromLong(2);
  pValue = PyObject_GetItem(results, index);
  if(!pValue) {
    cout << "Broken python code in eval, exiting" << endl;
    PyErr_Print();
    cout << "No addReturn returned" << endl;
    exit(-1);
  }
  temp = PyObject_GetItem(pValue, index_0);
  run.addReturn = PyInt_AsLong(temp);

  Py_DECREF(pValue);
  Py_DECREF(temp);
  Py_DECREF(index);

  index = PyInt_FromLong(3);
  pValue = PyObject_GetItem(results, index);
  if(!pValue) {
    cout << "Broken python code in eval, exiting" << endl;
    PyErr_Print();
    cout << "No cost returned" << endl;
    exit(-1);
  }
  temp = PyObject_GetItem(pValue, index_0);
  run.cost = PyInt_AsLong(pValue);

  Py_DECREF(pValue);
  Py_DECREF(temp);
  Py_DECREF(index);
  Py_DECREF(index_0);
  Py_DECREF(results);
  Py_DECREF(fitness_result);
  //FINALLY CLEAN

  if(add) {
    //We evaluated it, now add to all our surrogate models
    num_iterations++;
    for(int i = 0; i < dimension; i++) cout << x[i] << " ";
    cout << "fitness: " << run.fitness << " code: " << run.label << endl;
    sg->add(x, run.fitness - 200, 2 - (run.label == 0), run.addReturn);
    if(run.label == 0 ) {
      valid_set.push_back(x);
      if(run.fitness < best_fitness) {
        best_fitness = run.fitness;
        best_particle = x;
      }
    }
    training.push_back(x);
  } else {
    //About to run, stick on running vector
    run.is_finished = false;
    run.data = x;
    run.pos = mu_means.size() - 1;
    running.push_back(run);
    sg_cost->add(x, run.cost);
  }
}

void EGO::update_running(const long int &t)
{
  long int time = t;
  if(time == -1L) {
    time = 1000000000000000L;
    for(vector<struct running_node>::iterator node = running.begin(); node != running.end(); node++) {
      time = min(time, (long) node->cost);
    }
    time++;
  }
  if(time > 0) {
    for(vector<struct running_node>::iterator node = running.begin(); node != running.end();) {
      node->cost -= time;
      if(node->cost <= 0) {
        python_eval(node->data, true);

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
    update_time(time);
  }
  lambda = num_lambda - running.size();
}

void EGO::update_time(const long int &t)
{
  total_time += t;
}

EGO::EGO(int dim, Surrogate *s, vector<double> low, vector<double> up, string python_file_name, int search)
{
  dimension = dim;
  upper = up;
  lower = low;
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
  best_fitness = 10000000000;
  max_fitness = 0;
  total_time = 0L;
  is_discrete = false;
  is_new_result = false;
  use_brute_search = true;
  suppress = false;
  at_optimum = false;
  sg_cost = new Surrogate(dim, SEard);
  search_type = search;

  // Initialize the Python Interpreter
  Py_Initialize();
  cout << "Python initialised" << endl;
  string sys_append = "sys.path.append(\"";
  sys_append += python_file_name;
  sys_append += "\")\n";
  PyRun_SimpleString("import sys\n");
  PyRun_SimpleString(sys_append.c_str());

  // Build the name object
  const char *file_name = "fitness_script";
  pName = PyString_FromString(file_name);
  if(!pName) {
    cout << "Couldn't create pName for " << file_name << endl;
    exit(-1);
  }
  cout << "Got pName: " << PyString_AsString(pName) << endl;

  // Load the module object
  pModule = PyImport_Import(pName);
  if(!pModule) {
    cout << "Couldn't find module " << file_name << endl;
    exit(-1);
  }
  cout << "Got pModule" << endl;

  char name[] = "fitnessFunc";
  pFunc = PyObject_GetAttrString(pModule, name);
  if(!pFunc) {
    cout << "Couldn't find pFunc" << endl;
    exit(-1);
  }
  cout << "Got pFunc" << endl;
  if(!PyCallable_Check(pFunc)) {
    cout << "pFunc not callable" << endl;
    exit(-1);
  }

  Py_DECREF(pName);
  cout << "Got rid of pName" << endl;
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
  sg->add(x, y, label);
  if(label == 1 && y < best_fitness) {
    best_fitness = y;
    best_particle = x;
  }
}

vector<double> EGO::max_ei_par(int llambda) 
{
  vector<double> best;
  //if(llambda == 1) {
  //  vector<double> *x = brute_search_swarm(num_points, 1);
  //  if(x) {
  //    best = *x;
  //  } else {
  //    if(!suppress) cout << "Locally ";
  //    best = brute_search_local_swarm(best_particle, 1, 1, true);
  //  }
  //} else {
    if(search_type == 1) {
      vector<double> *ptr = brute_search_swarm(num_points, llambda);
      if(ptr) { 
        best = *ptr;
        delete ptr;
      } else {
        if(!suppress) cout << "Couldn't find new particles, searching in region of best" << endl;
	if(lambda == 1) {
          best = brute_search_local_swarm(best_particle, llambda, llambda, true, true);
	} else {
          best = brute_search_local_swarm(best_particle, llambda, llambda, true);
	}
        if(!suppress) {
          for(int i = 0; i < llambda * dimension; i++) {
            cout << best[i] << " ";
          }
          cout << " got around best" << endl;
        }
      }
    } else if(search_type == 2){ 
      int size = dimension * llambda;
      vector<double> low(size, 0.0), up(size, 0.0), x(size, 0.0);
      random_device rd;
      mt19937 gen(rd());

      for(int i = 0; i < size; i++) {
        low[i] = lower[i % dimension];
        up[i] = upper[i % dimension];
        x[i] = best_particle[i % dimension];
      }

      opt *op = new opt(size, up, low, this, is_discrete);
      best = op->swarm_optimise(x, pso_gen * size, population_size, 200);
      double best_fitness = op->best_part->best_fitness;

      if(!suppress) {
        cout << "[";
        for(int i = 0; i < llambda; i++) {
          for(int j = 0; j < dimension; j++) {
            cout << best[i*dimension + j] << " ";
          }
          cout << "\b; ";
        }
        cout << "\b\b] = best = "  << best_fitness << endl;
      }
      delete op;
    } else if(search_type == 3) {
      cout << "Not implemented" << endl;
      exit(-1);
    }
  //}

  iter++;
  return best;
}

void EGO::sample_plan(size_t F, int D)
{
  size_t size_latin = floor(F/3);
  cout << size_latin << "size" << endl;
  int* latin = ihs(dimension, size_latin, D, D);
  if(!latin) {
    cout << "Sample plan broke horribly, exiting" << endl;
    exit(-1);
  }
  cout << "Latin eval" << endl;
  for(size_t i = 0; i < size_latin; i++) {
    vector<double> x(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      double frac = (upper[j] - lower[j]) / (F-1);
      x[j] = lower[j] + frac*(latin[i*dimension+j]-1);
      if(is_discrete) {
        x[j] = floor(x[j]);
      }
    }
    while(running.size() == num_lambda) {
      //std::this_thread::sleep_for(std::chrono::milliseconds(20));
      //check_running_tasks();
      update_running();
    }
    //evaluate(x);
    python_eval(x);
  }
  sg->choose_svm_param(5, true);
  delete latin;

  //while(training.size() < F) {
  //  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  //  check_running_tasks();
  //}

  //long long int random_count = 1;
  //while(training.size() < F) {
  //  vector<double> point(dimension, 0.0);
  //  for(int j = 0; j < dimension; j++) {
  //    point[j] = round(uni_dist(lower[j], upper[j]));
  //    point[j] = min(point[j], upper[j]);
  //    point[j] = max(point[j], lower[j]);
  //  }

  //  if(sg->svm_label(&point[0]) != 1) {
  //    if(random_count % 1000 == 0) {
  //      cout << "Evaluating random permuation of valid" << endl;
  //      int choice = 0, valid_size = valid_set.size(), dist = 2;
  //      if(valid_size > 1) {
  //        choice = round(uni_dist(0, valid_size-1));
  //      }
  //      int restarts = 0;
  //      for(int j = 0; j < dimension; j++) {
  //        point[j] = valid_set[choice][j] + round(uni_dist(0, dist) - dist / 2);
  //        point[j] = min(point[j], upper[j]);
  //        point[j] = max(point[j], lower[j]);
  //        if(point[j] < lower[j] || point[j] > upper[j]) {
  //          j = -1;
  //        }
  //        if(j == dimension - 1) {
  //          if(!not_run(&point[0]) || !not_running(&point[0])) {
  //            j = -1;
  //            if(++restarts > 100) {
  //      	restarts = 0;
  //              dist++;
  //            }
  //          } else if(sg->svm_label(&point[0]) != 1) {
  //            if(++restarts < 100) {
  //              j = -1;
  //            }
  //          }
  //        } 
  //      }
  //      //dist = max(dist+1, 4);
  //      python_eval(point);
  //      sg->choose_svm_param(5);
  //    }
  //  } else {
  //    python_eval(point);
  //    sg->choose_svm_param(5);
  //  }
  //  random_count++;

  //  while(running.size() >= num_lambda) {
  //    update_running();
  //  }

  //}
  cout << "Adding extra permutations" << endl;
  for(size_t i = 0; i < F - size_latin; i++) {
    vector<double> point(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      point[j] = round(uni_dist(lower[j], upper[j]));
      point[j] = min(point[j], upper[j]);
      point[j] = max(point[j], lower[j]);
    }
    if(sg->svm_label(&point[0]) != 1) {
      int choice = 0, valid_size = valid_set.size(), radius = 2;
      if(valid_size > 1) {
        choice = round(uni_dist(0, valid_size-1));
      }
      int loops = 0;
      for(int j = 0; j < dimension; j++, loops++) {
	//int dist = floor((upper[j] - lower[j]) / radius);
        //point[j] = valid_set[choice][j] + round(uni_dist(0, dist) - dist / 2);
        point[j] = valid_set[choice][j] + round(uni_dist(0, radius) - radius / 2);
        if((point[j] > upper[j]) || (point[j] < lower[j])) {
	  j = -1; // reset loop
	}
        if(j == dimension - 1) {
          if(!not_run(&point[0]) || !not_running(&point[0])) {
            j = -1;
          }
        } 
	if(loops > 1000) {
	  radius++;
	  cout << loops << endl;
	}
      }
    }
    if(running.size() == num_lambda) {
      update_running();
    }

    python_eval(point);
    sg->choose_svm_param(5);
  }
  while(running.size() >= num_lambda) {
    update_running();
  }
  double x[3];
  x[2] = 32;
  for(int i = 1; i < 5; i++) {
    x[0] = i;
    for(int j = 40; j < 54; j++) {
      x[1] = j;
      cout << i << " "<<j<< " label:" <<sg->svm_label(x) << endl;
    }
  }
}

vector<double> EGO::local_random(const vector<double> &particle, double radius)
{
  int npts_plus[dimension + 1];
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      npts_plus[i] = pow(2*radius + 1, dimension - i);
    }
  }
  npts_plus[dimension] = 1;
  vector<double> x(dimension, 0.0);
  while(true) {
    for(int k = 0; k < 100; k++) {
      int i = round(uni_dist(0, npts_plus[0]));
      bool can_run = true;
      for(int j = 0; j < dimension; j++) {
        x[j] = particle[j] + floor(((i % npts_plus[j]) / npts_plus[j+1]) - radius);
        if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
      }
      if(can_run && not_run(&x[0]) && not_running(&x[0])) {
        return x;
      }
    }
    radius++;
  }
}

vector<double> *EGO::brute_search_swarm(int npts, int llambda)
{
  double best = -0.0;
  int size = dimension * llambda;
  vector<double> *best_point = new vector<double>(size, 0);
  unsigned long long npts_plus[dimension + 1];
  //unsigned long long loop[llambda];
  double steps[dimension];
  bool has_result = false;
  //for(int i = 0; i < llambda; i++) loop[i] = i;
  npts_plus[dimension] = 1;
  if(exhaustive) {
    for(int i = dimension - 1; i >= 0; i--) {
      steps[i] = 1.0;
      npts_plus[i] = (upper[i] - lower[i] + 1) * npts_plus[i+1];
    }
  } else {
    for(int i = 0; i < dimension; i++) {
      if(is_discrete) {
        steps[i] = (int) floor((upper[i] - lower[i]) / npts);
        if(steps[i] == 0) steps[i] = 1.0;
      } else {
        steps[i] = (upper[i] - lower[i]) / npts;
      }
      npts_plus[i] = (int) pow(npts + 1, dimension - i);
    }
  }

  if(llambda == 1) {
    vector<double> x(size, 0.0);
    for(unsigned long long i = 0; i < npts_plus[0]; i++) {
      bool can_run = true;
      for(int j = 0; j < dimension; j++) {
        x[j] = lower[j] + floor((i % npts_plus[j]) / npts_plus[j+1]) * steps[j];
        if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
      }

        if(can_run && not_run(&x[0]) && not_running(&x[0])) {
        pair<double, double> p = sg->predict(&x[0]);
        double result = -ei(p.first, p.second, best_fitness);
	//double cost = sg_cost->mean(&x[0]);
	//if(cost > 0) result /= cost;
        if(result < best) {
          best_point->assign(x.begin(), x.end());
          best = result;
          has_result = true;
        }
      }
    }
    if(!has_result) {
      best = 100000;
      for(unsigned long long i = 0; i < npts_plus[0]; i++) {
        bool can_run = true;
        for(int j = 0; j < dimension; j++) {
          x[j] = lower[j] + floor((i % npts_plus[j]) / npts_plus[j+1]) * steps[j];
          if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
        }

          if(can_run && not_run(&x[0]) && not_running(&x[0])) {
          pair<double, double> p = sg->predict(&x[0]);
          if(p.first < best) {
            best_point->assign(x.begin(), x.end());
            best = p.first;
            has_result = true;
          }
        }
      }
    }
  } else {
    double llambda_means[llambda];
    double llambda_vars[llambda];
    pair<double, double> best_mean_var;
    for(int i = 0; i < llambda; i++) {
      best = -0.0;
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

        if(can_run && not_run(&point[i*dimension]) && not_running(&point[i*dimension])) {
	  double result = 0.0;
	  pair<double, double> p = sg->predict(&point[i*dimension]);
	  if(i == 0) {
            result = -ei(p.first, p.second, best_fitness);
	  } else {
	    llambda_means[i] = p.first;
	    llambda_vars[i] = p.second;
            result = -1. * ei_multi(llambda_vars, llambda_means, i + 1, n_sims);
	  }
          if(result < best) {
            for(int k = 0; k < dimension; k++) {
              (*best_point)[i*dimension + k] = point[i*dimension + k];
            }
	    best_mean_var = p;
            best = result;
            if(i == llambda - 1) has_result = true;
	    found = true;
          }
        }
      }
      llambda_means[i] = best_mean_var.first;
      llambda_vars[i] = best_mean_var.second;
      if(!found) {
        delete best_point;
        return NULL;
      }
    }
  }

  if(has_result) {
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < llambda; i++) {
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


vector<double> EGO::brute_search_local_swarm(const vector<double> &particle, double radius, int llambda, bool has_to_run, bool random)
{
  double best = 1.0;
  int size = dimension * llambda;
  vector<double> best_point(size, 0);
  unsigned long long loop[llambda];
  double steps[dimension];
  unsigned long long npts_plus[dimension + 1];
  bool has_result = false;
  for(int i = 0; i < llambda; i++) loop[i] = i;
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

  if(llambda == 1) {
      vector<double> x(size, 0.0);
      for(unsigned long long i = 0; i < npts_plus[0]; i++) {
        bool can_run = true;
        for(int j = 0; j < dimension; j++) {
          x[j] = particle[j] + floor(((i % npts_plus[j]) / npts_plus[j+1]) - radius) * steps[j];
          if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
        }

        if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
	  if(random || sg->svm_label(&x[0]) == 1) {
            pair<double, double> p = sg->predict(&x[0]);
            double result = -ei(p.first, p.second, best_fitness);
            if(result < best) {
              best_point = x;
              best = result;
              has_result = true;
            }
	  }
        }
      }
  } else {
    for(int i = 0; i < llambda; i++) {
      best = 1.0;
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
	    if(random || sg->svm_label(&point[i*dimension]) == 1) {
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
              if(i == llambda - 1) has_result = true;
	      found = true;
            }
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
      if(radius / llambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
        at_optimum = true;
        return best_point;
      } else {
        //cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run);
      }
    } else {
      if(llambda == 1 && radius < 3) {
	max_points *= 2;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run);
      } else if(radius / llambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
        return brute_search_local_swarm(particle, 1, llambda, has_to_run, true);
      } else {
        cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run);
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
    double y_diff = y_min - y - 200;
    double y_diff_s = y_diff / s;
    return y_diff * phi(y_diff_s) + s * normal_pdf(y_diff_s);
  }
}

double EGO::ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n)
{
    double sum_ei=0.0, e_i=0.0;
    int max_mus = mu_means.size();

    for (int k=0; k < n; k++) {
        double min = best_fitness - 200;
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
