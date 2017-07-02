#include <vector>
#include <random>
#include "surrogate.hpp"
#include <mutex>
#include "evaluator.hpp"
#include <gsl/gsl_rng.h>
#pragma once

using namespace std;

class EGO
{
  public:
    //Constructor
    EGO(vector<pair<double, double>> boundaries, Evaluator& evaluator);
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
    vector<double> max_ei_par(int llambda);
    double fitness(const vector<double> &x);
    void evaluate(const vector<double> &x);
    void run();
    vector<double> best_result();
    vector<double> brute_search_local_swarm(const vector<double> &particle, double radius=1.0, int llambda=1, bool has_to_run=false, bool random=false, bool use_mean=false);
    vector<double>* brute_search_swarm(int npts=10, int llambda=1, bool use_mean=false);
    double ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n, double y_best);
    double ei(double y, double var, double y_min);
    void check_running_tasks();
    void worker_task(struct running_node& run);
    bool not_run(const double x[]);
    bool not_running(const double x[]);
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
    int search_type;
    vector<double> best_particle;
    long double best_fitness;
    bool is_discrete;
    bool use_brute_search;
    bool use_cost;
    bool suppress;
    bool exhaustive;

	Evaluator& evaluator;

    mutex running_mtx;
    vector<struct running_node> running;
    vector<vector<double>> training;
    vector<double> training_f;
    vector<vector<double>> valid_set;
    vector<double> mu_means;
    vector<double> mu_vars;
    vector<double> lower;
    vector<double> upper;

    Surrogate* sg;
    Surrogate* sg_cost;

	gsl_rng* rng;

    void python_eval(vector<double> &x, bool add=false);
    void run_quad();
    void update_running(const long int &t=-1l);
    bool has_run(const vector<double> &point);
    void latin_hypercube(size_t F, int D);
    vector<double> local_random(double radius=1.0, int llambda=1);

	void update_best_result(vector<double> x, double y);
};
