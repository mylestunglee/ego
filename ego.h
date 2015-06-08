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
    ~EGO();

    struct running_node {
      double fitness;
      bool is_finished;
      vector<double> data;
      int pos;
    };

    //Functions
    vector<double> max_ei_par(int lambda);
    double fitness(const vector<double> &x);
    void evaluate(const vector<double> &x);
    void run();
    vector<double> best_result();
    void add_training(const vector<double> &x, double y);
    vector<double> brute_search_local_swarm(const vector<double> &particle, double radius=1.0, int lambda=1, bool has_to_run=false);
    vector<double>* brute_search_swarm(int npts=10, int lambda=1);
    double ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n);
    double ei(double y, double S2, double y_min);
    void check_running_tasks();
    void worker_task(vector<double> data);
    bool not_run(double x[]);
    bool not_running(double x[]);
    void sample_plan(int F, int D=5);

    //Variables
    int dimension = 1;
    int n_sims = 50;
    int max_iterations = 1000;
    int num_iterations = 0;
    int num_lambda = 3;
    int population_size = 100;
    int num_points = 10;
    int max_points = 10;
    int pso_gen = 1;
    int iter = 0;
    vector<double> best_particle;
    double best_fitness = 100000000;
    double max_fitness = 0;
    bool is_discrete = false;
    bool is_new_result = false;
    bool use_brute_search = false;
    bool swarm = false;
    bool suppress = false;
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

    Surrogate *sg;

};

