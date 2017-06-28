#include "ego.hpp"
#include <vector>
#include <random>

#pragma once

using namespace std;

class Particle
{
  public:
    vector<double> p;
    vector<double> speed;
    vector<double> best;
    double best_fitness;
    int group;
};

class opt 
{
  public:
    int dimension;
    int last_gen;
    vector<double> upper;
    vector<double> lower;
    bool is_discrete;
    EGO *ego;
    Particle *best_part;
    vector<Particle *> particles;

    //vector<std::uniform_real_distribution<double>> *space_generator;
    //vector<std::uniform_real_distribution<double>> *speed_generator;
    vector<double> speed_max;

    //Functions
    opt(int d, vector<double> u, vector<double> l, EGO *e, bool disc);
    ~opt();
    void update_particles(int generation, int max_iter);
    void filter();
    void generate(int pop, int groups=1);
    vector<double> swarm_optimise(int max_gen, int pop = 100, int min_gen=100);
    vector<double> swarm_optimise(vector<double> best, int max_gen, int pop, int min_gen=100);
    vector<double> swarm_main_optimise(int max_gen, int min_gen);
    vector<vector<double>> combined_optimise(vector<double> best, int max_gen, int pop, int num_groups);
};

double uni_dist(double N, double M);
