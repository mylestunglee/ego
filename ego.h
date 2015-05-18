#include <vector>
#include <random>
#include "surrogate.h"
#pragma once

using namespace std;

class EGO
{
  public:
    //Constructor
    EGO(int dim, vector<double> low, vector<double> up, double(*fit)(double x[]));

    //Functions
    vector<double> max_ei_par(int lambda);
    double ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n);
    double fitness(vector<double> x);
    double (* proper_fitness) (double x[]);
    void fitness_function(double x[]);
    void run();

    //Variables
    int dimension = 1;
    int n_sims = 50;
    int max_iterations = 50;
    int num_lambda = 1;
    vector<vector<double>> running;
    vector<pair<vector<double>, double>> training;

    vector<double> mu_means;
    vector<double> mu_vars;
    vector<double> lower;
    vector<double> upper;
    vector<uniform_real_distribution<>> _generator;

    Surrogate *sg;

};


