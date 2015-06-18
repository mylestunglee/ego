#include "ego.h"
#include "optimise.h"
#include "ihs.hpp"
#include <ctime>

//#define _GLIBCXX_USE_NANOSLEEP //Need to send thread to sleep on old GCC

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
    delete sg_cost_soft;
  }
}

void EGO::run_quad()
{
  if(!suppress) cout << "Started, dim=" << dimension << ", lambda=" <<
  num_lambda << " max_f=" << max_fitness << endl;
  sg->train_gp_first();
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing

    //auto t1 = std::chrono::high_resolution_clock::now();
    std::clock_t t3 = std::clock();
    sg->train();
    sg_cost->train();
    if(train_cost_soft && !sg_cost_soft->gp_is_trained) {
      sg_cost_soft->train_gp_first();
    } else {
      sg_cost_soft->train();
    }
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto t3 = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
    t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
    update_running(t3);

    if(is_max) {
      if(at_optimum || best_fitness >= max_fitness) {
        if(!suppress) {
          cout << "Found best at [";
          for(int i = 0; i < dimension; i++) {
            cout << best_particle[i] << ", ";
          }
          cout << "\b\b] with fitness [" << best_fitness << "]" << endl;
        }
        return;
      }
    } else {
      if(at_optimum || best_fitness <= max_fitness) {
        if(!suppress) {
          cout << "Found best at [";
          for(int i = 0; i < dimension; i++) {
            cout << best_particle[i] << ", ";
          }
          cout << "\b\b] with fitness [" << best_fitness << "]" << endl;
        }
        return;
      }
    }


    if(is_new_result && !suppress) {
      cout << "Iter: " << num_iterations << " / " << max_iterations;
      cout << ", RUNNING: " << running.size() << " Lambda: " << lambda;
      cout << " best " << best_fitness << endl;
      is_new_result = false;
    }

    if(lambda > 0) {
      int temp_lambda = lambda;
      //t1 = std::chrono::high_resolution_clock::now();
      t3 = std::clock();
      vector<double> best_xs = max_ei_par(temp_lambda);
      //t2 = std::chrono::high_resolution_clock::now();
      //t3 = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
      t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
      update_running(t3);

      t3 = std::clock();
      for(int l = 0; l < temp_lambda; l++) {
        vector<double> y(dimension, 0.0);
        for(int j = 0; j < dimension; j++) {
          y[j] = best_xs[l * dimension + j];
        }

        if(!at_optimum && not_running(&y[0]) && not_run(&y[0])) {
          python_eval(y);
        } else {
	  if(!suppress) {
            cout << "Have run: ";
            for(int j = 0; j < dimension; j++) {
              cout << y[j] << " ";
            }
	  }
          y = local_random();
	  int s = y.size();
	  if(s != dimension || at_optimum) {
	    break;
	  }
          python_eval(y);
        }
      }
      t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
      update_running(t3);
    }
    if(lambda == 0) update_running();
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

    pair<double, double> p = sg->predict(data, true);
    mu_means.push_back(p.first);
    mu_vars.push_back(p.second);
  }

  struct running_node run; 
  PyObject *args, *index, *args2, *fitness_result, *results, *temp, *index_0;
  args = PyList_New(x.size());
  //Set up arguments for call to fitness function
  for(unsigned int i = 0; i < x.size(); i++) {
    long val = round(x[i]);
    if(val > 0) {
      size_t v = val;
      pValue = PyInt_FromSize_t(v);
    } else {
      pValue = PyInt_FromLong(val);
    }
    if(pValue == NULL || PyErr_Occurred()) {
      cout << "Broken python code in eval, exiting" << endl;
      cout << "Broke on adding " << x[i] << " to arg list" << endl;
      PyErr_Print();
      Py_DECREF(pValue);
      Py_DECREF(args);
      exit(-1);
    }
    PyList_SetItem(args, i, pValue);
    //cout << "Break on pValue " << pValue << " for " <<val <<endl;
    if(pValue) {
      //Python breaks horribly for no reason, so we are fudging this and leaking
      //memory everywhere
      //Py_XDECREF(pValue);
    }
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
    args = NULL;
  }
  if(args2)  {
    Py_DECREF(args2);
    args2 = NULL;
  }
  if(pState) {
    Py_DECREF(pState);
    pState = NULL;
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
  run.true_cost = run.cost;
  run.cost = abs(run.cost);

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
    int label = 2 - (int) (run.label == 0); //1 if run.label == 0
    sg->add(x, run.fitness, label, run.addReturn);
    if(run.addReturn == 0) {
      if(run.true_cost < 0) {
	train_cost_soft = true;
        sg_cost_soft->add(x, -run.true_cost);
	cout << "Software cost " << -run.true_cost << endl;
      } else {
        sg_cost->add(x, run.true_cost);
      }
    }
    if(run.label == 0 ) {
      valid_set.push_back(x);
      if(is_max) {
        if(run.fitness > best_fitness) {
          best_fitness = run.fitness;
          best_particle = x;
        }
      } else {
        if(run.fitness < best_fitness) {
          best_fitness = run.fitness;
          best_particle = x;
        }
      }
      training_f.push_back(run.fitness);
    }
    training.push_back(x);
  } else {
    //About to run, stick on running vector
    run.is_finished = false;
    run.data = x;
    run.pos = mu_means.size() - 1;
    running.push_back(run);
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

void EGO::update_time(long int t)
{
  total_time += t;
  cout << "Total time taken is " << total_time << endl;
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
  total_time = 0;
  is_discrete = false;
  is_new_result = false;
  use_brute_search = true;
  suppress = false;
  at_optimum = false;
  use_cost = false;
  train_cost_soft = false;
  is_max = false;
  sg_cost = new Surrogate(dim, SEard);
  sg_cost_soft = new Surrogate(dim, SEard);
  search_type = search;

  // Initialize the Python Interpreter
  Py_Initialize();
  cout << "Python initialised" << endl;
  string sys_append = "sys.path.append(\"";
  sys_append += python_file_name;
  sys_append += "\")\n";
  PyRun_SimpleString("import sys\n");
  PyRun_SimpleString(sys_append.c_str());
  cout << python_file_name << endl;

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
  population_size = 200;
  num_points = 10;
  max_points = 10;
  pso_gen = 1;
  iter = 0;
  total_time = 0;
  best_fitness = 100000000;
  max_fitness = 0;
  is_discrete = false;
  is_new_result = false;
  use_brute_search = true;
  suppress = false;
  at_optimum = false;
  use_cost = false;
  pName = NULL;
  pFunc = NULL;
}

void EGO::run()
{
  if(!suppress) cout << "Started, dim=" << dimension << ", lambda=" << num_lambda << endl;
  sg->train_gp_first();
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing
    std::clock_t t3 = std::clock();
    check_running_tasks();
    sg->train();
    //t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
    //update_time(t3);

    if(at_optimum || best_fitness <= max_fitness) {
      if(!suppress) {
        cout << "Found best at [";
        for(int i = 0; i < dimension; i++) {
          cout << best_particle[i] << ", ";
        }
        cout << "\b\b] with fitness [" << best_fitness << "]" << endl;
      }
      while(running.size() > 0){
        //std::this_thread::sleep_for(std::chrono::milliseconds(50));
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

    //t3 = std::clock();
    if(lambda > 0) {
      cout << " max " <<endl;
      vector<double> best_xs = max_ei_par(lambda);
      cout << " max " <<endl;
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
    t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
    cout << "Lambda" << lambda << endl;
    update_time(t3);
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
    pair<double, double> p = sg->predict(data, true);
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
  //std::thread (&EGO::worker_task, this, x).detach();
  
  worker_task(x);
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
      if(sg->is_svm) {
        add_training(node->data, node->fitness, node->label);
      } else {
	training.push_back(node->data);
        sg->add(node->data, node->fitness);
	if(is_max) {
	  if(node->fitness > best_fitness) {
	    best_particle = node->data;
	    best_fitness = node->fitness;
	  }
	} else {
	  if(node->fitness < best_fitness) {
	    best_particle = node->data;
	    best_fitness = node->fitness;
	  }
	}
      }

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
  double cost = 0;

  for(int i = 0; i < num; i++) {
    double y [dimension];

    for(int j = 0; j < dimension; j++) {
      y[j] = x[i * dimension + j];
    }

    if(not_running(y)) {
      pair<double, double> p = sg->predict(y, true);
      lambda_means[i] = p.first;
      lambda_vars[i] = p.second;
      if(is_max && p.second == 0) {
        lambda_means[i] = -p.first;
      }
      //cout << p.first << " "<< p.second << endl;
      if(use_cost) {
        pair<double, double> p_cost = sg_cost->predict(y);
        pair<double, double> s_cost = sg_cost_soft->predict(y);
	cost += p_cost.first + s_cost.first;
      }
    } else {
      lambda_means[i] = 100000000000;
      lambda_vars[i] = 0;
    }
  }

  double result = -1. * ei_multi(lambda_vars, lambda_means, num, n_sims, sg->best_raw());
  if(use_cost && cost > 0) {
    result *= 10;
    result /= cost;
  }
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
  static bool use_mean = false;
  if(search_type == 1) {
    vector<double> *ptr = brute_search_swarm(num_points, llambda, use_mean);
    if(ptr) { 
      best = *ptr;
      delete ptr;
    } else {
      if(!suppress) cout << "Couldn't find new particles, searching in region of best" << endl;
      best = local_random(1.0, llambda);
      if(!suppress && !at_optimum) {
        for(size_t i = 0; i < best.size(); i++) {
          cout << best[i] << " ";
        }
        cout << " got around best" << endl;
      }
    }
  } else if(search_type == 2){ 
    int size = dimension * llambda;
    vector<double> low(size, 0.0), up(size, 0.0), x(size, 0.0);

    for(int i = 0; i < size; i++) {
      low[i] = lower[i % dimension];
      up[i] = upper[i % dimension];
      x[i] = best_particle[i % dimension];
    }

    opt *op = new opt(size, up, low, this, is_discrete);
    best = op->swarm_optimise(x, pso_gen, population_size * size);
    double best_f = op->best_part->best_fitness;

    if(best_f >= -0.00000001) {
      best = local_random(1.0, llambda);
    } 
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < llambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << best[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best_f << endl;
    }
    delete op;
   } else {
    int size = dimension * llambda;
    vector<double> low(size, 0.0), up(size, 0.0), x(size, 0.0);

    for(int i = 0; i < size; i++) {
      low[i] = lower[i % dimension];
      up[i] = upper[i % dimension];
      x[i] = best_particle[i % dimension];
    }

    opt *op = new opt(size, up, low, this, is_discrete);
    vector<vector<double>> swarms = op->combined_optimise(x, pso_gen, population_size * size, lambda);
    double best_f = 0.0;
    if(swarms.size() <= llambda) {
      best = local_random(1.0, llambda);
    } else {
      for(size_t i = 0; i < swarms[llambda].size(); i++) {
        best_f = min(best_f, swarms[lambda][i]);
      }
      if(best_f >= -0.000000001) {
        best = local_random(1.0, llambda);
        best_f = fitness(best);
      } else {
        int old_n_sims = n_sims;
        n_sims *= 10;
	vector<double> x, y;
        for(int i = 0; i < llambda; i++) {
	  x = swarms[i];
          y = brute_search_local_swarm(x, 1, 1, true, false, use_mean);
	  if(y.size() < size) {
	    y = local_random();
	  }
          for(int j = 0; j < dimension; j++) {
            best.push_back(y[j]);
          }
        }
	x.clear();
	y.clear();
        best_f = fitness(best);
        n_sims = old_n_sims;
      }
    }
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < llambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << best[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best_f << endl;
    }
    delete op;
  } 

  iter++;
  return best;
}

void EGO::latin_hypercube(size_t F, int D) 
{
  int* latin = ihs(dimension, F, D, D);
  if(!latin) {
    cout << "Sample plan broke horribly, exiting" << endl;
    exit(-1);
  }
  for(size_t i = 0; i < F; i++) {
    vector<double> x(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      double frac = (upper[j] - lower[j]) / (F-1);
      x[j] = lower[j] + frac*(latin[i*dimension+j]-1);
      if(is_discrete) {
        x[j] = floor(x[j]);
      }
    }
    while(running.size() >= num_lambda) {
      //std::this_thread::sleep_for(std::chrono::milliseconds(20));
      check_running_tasks();
    }
    evaluate(x);
  }
}

void EGO::sample_plan(size_t F, int D)
{
  size_t size_latin = floor(F/3);
  cout << size_latin << " size" << endl;
  D += rand() % 5;
  int* latin = ihs(dimension, size_latin, D, D);
  if(!latin) {
    cout << "Sample plan broke horribly, exiting" << endl;
    exit(-1);
  }
  cout << "Latin eval" << endl;
  for(size_t i = 0; i < size_latin; i++) {
    vector<double> x(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      double frac = (upper[j] - lower[j]) / (size_latin-1);
      x[j] = lower[j] + frac*(latin[i*dimension+j]-1);
      if(is_discrete) {
        x[j] = floor(x[j]);
      }
    }
    while(running.size() >= num_lambda) {
      //std::this_thread::sleep_for(std::chrono::milliseconds(20));
      //check_running_tasks();
      update_running();
    }
    //evaluate(x);
    python_eval(x);
  }
  while(valid_set.size() < 1 || running.size() >= num_lambda) {
    update_running();
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
    for(int j = 0, loops = 0; j < dimension; loops++, j++) {
      int rand_num = rand() % ((int)round(upper[j] - lower[j]) + 1);
      point[j] = lower[j] + rand_num;
      point[j] = min(point[j], upper[j]);
      point[j] = max(point[j], lower[j]);
      if(j == dimension - 1) {
        if(loops < 1000 && (sg->svm_label(&point[0]) != 1 || has_run(point))) {
	  j = -1; // reset and try and find another random point
	}
      }
    }
    if(sg->svm_label(&point[0]) != 1 || has_run(point)) {
      int choice = 0, valid_size = valid_set.size(), radius = 2;
      if(valid_size > 1) {
        choice = rand() % valid_size;
      }
      int loops = 0;
      for(int j = 0; j < dimension; j++, loops++) {
	//int dist = floor((upper[j] - lower[j]) / radius);
        //point[j] = valid_set[choice][j] + round(uni_dist(0, dist) - dist / 2);
        int rand_num = rand() % (radius + 1);
        point[j] = valid_set[choice][j] + (rand_num - radius / 2);
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
    cout << "svm label: " << sg->svm_label(&point[0]) << endl;
    python_eval(point);
    sg->choose_svm_param(5);

    while(running.size() >= num_lambda) {
      update_running();
    }

  }
  while(running.size() >= num_lambda) {
    update_running();
  }
}

vector<double> EGO::local_random(double radius, int llambda)
{
  if(!suppress) {
    cout << "Evaluating random pertubations of best results" << endl;
  }
  double noise = sg->error();
  vector<int> index;
  vector<vector<double>> points;
  for(size_t i = 0; i < valid_set.size(); i++) {
    if(abs(training_f[i] - best_fitness) < noise) {
      index.push_back(i);
    }
  }
  vector<double> low_bounds(dimension, -10000000.0);
  double up;
  vector<double> x(dimension, 0.0);
  int num[dimension+1];
  for(size_t i = 0; i < index.size(); i++) {
    num[0] = 1;
    for(int j = 0; j < dimension; j++) {
      double point = valid_set[index[i]][j];
      low_bounds[j] = max(lower[j], point - radius);
      up = min(upper[j], point + radius);
      num[j+1] = num[j] * (int) (up - low_bounds[j] + 1);
      //cout << j << " " <<num[j+1] <<" "<<low_bounds[j]<<" "<<up<< " possible" << endl;
    }
    for(int j = 0; j < num[dimension]; j++) {
      for(int k = 0; k < dimension; k++) {
        x[k] = low_bounds[k] + (j % num[k+1]) / num[k];
	//cout << x[k] << " ";
      }
      //cout << endl;
      if(!has_run(x)) {
	bool can_run = true;
        for(size_t z = 0; z < points.size(); z++) {
	  for(int l = 0; l < dimension; l++) {
	    if(x[l] != points[z][l]) {
	      break;
	    } else if(l == dimension - 1) {
	      can_run = false;
	      break;
	    }
	  }
	  if(!can_run) {
	    break;
	  }
        }
	if(can_run) {
	  points.push_back(x);
	  //cout << "able" << endl;
        }
      }
    }
  }
  int size = points.size();
  if(size > 0) {
    vector<double> result(dimension * llambda, 0.0);
    double choices[llambda];
    for(int i = 0; i < llambda; i++) {
      choices[i] = -1;
      int ind = rand() % size;
      if(size >= llambda) {
        for(int j = 0; j < llambda; j++) {
          if(ind == choices[j]) {
            j = -1;
            ind = rand() % size;
          }
        }
      }
      for(int j = 0; j < dimension; j++) {
        result[i*dimension+j] = points[ind][j];
      }
    }
    return result;
  } else {
    points.clear();
    low_bounds.clear();
    index.clear();
    x.clear();
    return local_random(radius + 1, llambda);
  }
}

vector<double> *EGO::brute_search_swarm(int npts, int llambda, bool use_mean)
{
  double best = -0.0;
  double y_best = sg->best_raw();
  int size = dimension * llambda;
  vector<double> *best_point = new vector<double>(size, 0);
  unsigned long long npts_plus[dimension + 1];
  unsigned long long loop[llambda];
  double steps[dimension];
  bool has_result = false;
  for(int i = 0; i < llambda; i++) loop[i] = i;
  npts_plus[dimension] = 1;
  if(is_discrete && exhaustive) {
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
        if(sg->svm_label(&x[0]) == 1) {
          double result = fitness(x);
          if(result < best) {
            best_point->assign(x.begin(), x.end());
            best = result;
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
      if(use_mean) best = 1000000;
      vector<double> point((i+1)*dimension, 0.0);
      bool found = false;

      for(int j = 0; j < i*dimension; j++) {
        point[j] = (*best_point)[j];
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
          point[i * dimension + k] = lower[k] + floor((j % npts_plus[k]) / npts_plus[k+1]) * steps[k];
          if(point[i * dimension + k] > upper[k] || point[i * dimension + k] < lower[k]) can_run = false;
        }

        if(can_run && not_run(&point[i*dimension]) && not_running(&point[i*dimension])) {
	  if(sg->svm_label(&point[i*dimension]) == 1) {
	    pair<double, double> p = sg->predict(&point[i*dimension], true);
	    double result = 0.0;
	    if(use_mean) {
	      result = sg->mean(&point[i*dimension]);
	    } else if(use_cost) {
	      result = fitness(point);
	    } else {
	      llambda_means[i] = p.first;
	      llambda_vars[i] = p.second;
              result = -1. * ei_multi(llambda_vars, llambda_means, i + 1, n_sims, y_best);
	    }
            if(result < best) {
              for(int k = 0; k < dimension; k++) {
                (*best_point)[i*dimension + k] = point[i*dimension + k];
              }
	      best_mean_var = p;
              best = result;
	      loop[i] = j;
              if(i == llambda - 1) has_result = true;
	      found = true;
            }
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


vector<double> EGO::brute_search_local_swarm(const vector<double> &particle, double radius, int llambda, bool has_to_run, bool random, bool use_mean)
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
            double result = fitness(x);
	    if(num_lambda == 1){
	      pair<double, double> p = sg->predict(&x[0], true);
	      result = -ei(p.first, p.second, sg->best_raw());
	    }
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
	    double result = fitness(point);
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
      if(llambda == 1) {
        if(radius < 3) {
          return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
        } else {
          if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
	  best_point.clear();
          return best_point;
	}
      }
      if(radius / llambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
	best_point.clear();
        return best_point;
      } else {
        //cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
      }
    } else {
      if(llambda == 1 && radius < 3) {
	max_points *= 2;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
      } else if(radius / llambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
	best_point.clear();
        return best_point;
      } else {
        cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
      }
    }
  }
}

bool EGO::has_run(const vector<double> &point)
{
  return !(not_run(&point[0]) && not_running(&point[0]));
}

bool EGO::not_run(const double x[])
{
  static double eps = 0.001;
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

bool EGO::not_running(const double x[])
{
  static double eps = 0.001;
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
    if(is_max) {
      double y_diff = y - y_min;
      double y_diff_s = y_diff / s;
      return y_diff * phi(y_diff_s) + s * normal_pdf(y_diff_s);
    } else {
      double y_diff = y_min - y;
      double y_diff_s = y_diff / s;
      return y_diff * phi(y_diff_s) + s * normal_pdf(y_diff_s);
    }
  }
}

double EGO::ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n, double y_best)
{
    double sum_ei=0.0, e_i=0.0;
    int max_mus = mu_means.size();

    if(is_max) {
      for (int k=0; k < n; k++) {
          double max = y_best;
          for(int i=0; i < max_mus; i++){
              double mius = gaussrand1()*mu_vars[i] + mu_means[i];
              if (mius > max)
                  max = mius;
          }
          double max2=-100000000.0;
          for(int j=0;j<max_lambdas;j++){
              double l = gaussrand1()*lambda_s2[j] + lambda_mean[j];
              if (l > max2) {
                  max2 = l;
	      }
          }
          
          e_i = max2 - max;
          if (e_i < 0.0) {
            e_i = 0.0;
          }
          sum_ei = e_i + sum_ei;
      }
    } else { 
      for (int k=0; k < n; k++) {
          double min = y_best;
          for(int i=0; i < max_mus; i++){
              double mius = gaussrand1()*mu_vars[i] + mu_means[i];
              if (mius < min)
                  min = mius;
          }
          double min2=100000000.0;
          for(int j=0;j<max_lambdas;j++){
              double l = gaussrand1()*lambda_s2[j] + lambda_mean[j];
              if (l < min2){
                  min2 = l;
	      }
          }
          
          e_i = min - min2;
          if (e_i < 0.0) {
            e_i = 0.0;
          }
          sum_ei = e_i + sum_ei;
      }
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
