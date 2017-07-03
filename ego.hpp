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
    vector<double> best_result();
    vector<double> brute_search_local_swarm(const vector<double> &particle, double radius=1.0, int llambda=1, bool has_to_run=false, bool random=false, bool use_mean=false);
    vector<double>* brute_search_swarm(int npts=10, int llambda=1, bool use_mean=false);
    double ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n, double y_best);
    double ei(double y, double var, double y_min);
    void check_running_tasks();
    bool not_run(const double x[]);
    bool not_running(const double x[]);
    void sample_plan(size_t n);
	void uniform_sample(size_t n);

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

    void run_quad();
    void update_running(const long int &t=-1l);
    bool has_run(const vector<double> &point);
    vector<double> local_random(double radius=1.0, int llambda=1);

	void update_best_result(vector<double> x, double y);

	static void evaluate2(EGO* ego, vector<double> x);
	void evaluate(vector<vector<double>> xs);
	vector<vector<double>> group(vector<double> xs, size_t size);
};
