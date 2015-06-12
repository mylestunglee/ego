#include "Python.h"
#include <vector>
#include "functions.h"
#include <random>
#include "surrogate.h"
#include <mutex>
#pragma once

using namespace std;

class EGO
{
  public:
    //Constructor
    EGO(int dim, Surrogate *s, vector<double> low, vector<double> up, double(*fit)(double x[]));
    ~EGO();

    struct running_node {
      double fitness;
      int label;
      bool is_finished;
      int addReturn;
      int cost;
      int pos;
      vector<double> data;
    };

    //Functions
    vector<double> max_ei_par(int lambda);
    double fitness(const vector<double> &x);
    void evaluate(const vector<double> &x);
    void run();
    vector<double> best_result();
    void add_training(const vector<double> &x, double y, int label);
    vector<double> brute_search_local_swarm(const vector<double> &particle, double radius=1.0, int lambda=1, bool has_to_run=false);
    vector<double>* brute_search_swarm(int npts=10, int lambda=1);
    double ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n);
    double ei(double y, double S2, double y_min);
    void check_running_tasks();
    void worker_task(vector<double> data);
    bool not_run(double x[]);
    bool not_running(double x[]);
    void sample_plan(size_t F, int D=5);

    //Variables
    int dimension;
    int n_sims;
    int max_iterations;
    int num_iterations;
    size_t num_lambda;
    int lambda;
    int population_size;
    int num_points;
    int max_points;
    int pso_gen;
    int iter;
    vector<double> best_particle;
    long double best_fitness;
    long double max_fitness;
    bool at_optimum;
    bool is_discrete;
    bool is_new_result;
    bool use_brute_search;
    bool suppress;
    vector<double> discrete_steps;

    PyObject *pName; 
    PyObject *pModule; 
    PyObject *pDict; 
    PyObject *pFunc; 
    PyObject *pValue; 
    PyObject *pState; 

    double (* proper_fitness) (double x[]);

    mutex running_mtx;
    vector<struct running_node> running;
    vector<vector<double>> training;
    vector<long double> training_fitness;
    vector<double> mu_means;
    vector<double> mu_vars;
    vector<double> lower;
    vector<double> upper;

    Surrogate *sg;
    Surrogate *sg_cost;

  EGO(int dim, Surrogate *s, vector<double> low, vector<double> up, string python_file_name);
  void python_eval(const vector<double> &x, bool add=false);
  void run_quad();

};

