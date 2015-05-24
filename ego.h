#include <vector>
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

    struct running_node {
      double fitness;
      bool is_finished;
      vector<double> data;
      int pos;
    };

    //Functions
    vector<double> max_ei_par(int lambda);
    double fitness(vector<double> x);
    void evaluate(vector<double> x);
    void run();
    vector<double> best_result();
    void add_training(vector<double> x, double y);
    vector<double> brute_search(int npts);
    double get_y_min();
    double ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n);
    double ei(double y, double S2, double y_min);
    void check_running_tasks();
    void worker_task(running_node *node);
    bool not_run(double x[]);

    //Variables
    int dimension = 1;
    int n_sims = 50;
    int max_iterations = 30;
    int num_iterations = 0;
    int num_lambda = 1;
    vector<double> best_particle;
    double best_fitness = 100000000;
    double max_fitness = 0;
    double min_running = 100000000;
    bool is_discrete = false;
    vector<double> discrete_steps;

    double (* proper_fitness) (double x[]);

    mutex running_mtx;
    vector<struct running_node> running;
    vector<vector<double>> training;
    vector<double> training_fitness;
    vector<double> mu_means;
    vector<double> mu_vars;
    vector<double> lower;
    vector<double> upper;
    vector<uniform_real_distribution<>> _generator;

    Surrogate *sg;

};

