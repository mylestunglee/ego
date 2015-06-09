#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>
#include <chrono>

using namespace std;
EGO *reset_ego();


EGO *reset_ego()
{
  //double (*fitness)(double x[]) =  &sphere_3;
  //int dimension = 3;
  //vector<double> lower = {-5.0, -5.0, -5.0};
  //vector<double> upper = {5.0, 5.0, 5.0};

  //double (*fitness)(double x[]) =  &sphere_3;
  //int dimension = 3;
  //vector<double> lower = {-13.0, -13.0, -13.0};
  //vector<double> upper = {13.0, 13.0, 13.0};

  double (*fitness)(double x[]) =  &sphere_20;
  int dimension = 10;
  vector<double> lower = {-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0 };
  vector<double> upper = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };

  //double (*fitness)(double x[]) =  &easy_test2;
  //int dimension = 1;
  //vector<double> lower = {-10.0, -13.0, -13.0};
  //vector<double> upper = {10.0, 13.0, 13.0};

  //double (*fitness)(double x[]) =  &sphere_4;
  //int dimension = 4;
  //vector<double> lower = {-5.0, -5.0, -5.0, -5.0};
  //vector<double> upper = {5.0, 5.0, 5.0, 5.0};
  
  //cout << "Building" << endl;
  Surrogate *sg(new Surrogate(dimension, SEiso, true));
  //sg->set_params(0.0, 0.0);
  EGO *ego(new EGO(dimension, sg, lower, upper, fitness));
  //cout << "Built" << endl;

  ego->suppress = true;
  ego->is_discrete = true;
  ego->n_sims = 10;
  ego->max_points = upper[0] - lower[0];
  ego->num_points = ego->max_points;
  ego->pso_gen = 100;

  //cout << "Sample"<<endl;
  ego->sample_plan(ego->max_points, 5);
  //cout << "Sampled"<<endl;
  return ego;
}

int main(int argc, char * argv[]) 
{
  EGO* ego = NULL;

  //ego = reset_ego();
  //ego->suppress = false;
  //ego->use_brute_search = true;
  //vector<double> x = {6};
  //ego->evaluate(x);
  //x = {4};
  //ego->evaluate(x);
  //x = {8};
  //ego->evaluate(x);
  //x = {-5};
  //ego->evaluate(x);
  //ego->check_running_tasks();
  //ego->sg->train();
  ////ego->run();
  ////ego->sg->train();
  //for(int i = 0; i < 10; i++) {
  //x[0] = i;
  //pair<double, double> p = ego->sg->predict(&x[0]);
  //cout << i << " " << ego->ei(p.first, p.second, ego->best_fitness) << " " << p.first << " " << p.second << " " << ego->sg->svm_label(&x[0]) << endl;
  //}

  //for(int i = 1; i < 3; i++) {
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  ego->brute_search_swarm(10, i);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute search npts=" << npts[i] << " l=2, took " << t3 << endl;
  //  ego = reset_ego();
  //}

  //for(int i = 2; i < 7; i++) {
  //  ego = reset_ego();
  //  ego->n_sims = i * 10 / 2;
  //  ego->n_sims = 10;
  //  ego->use_brute_search = false;
  //  ego->pso_gen = 500;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  vector<double> x = ego->max_ei_par(i);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "PSO search l=" << i << " nsims " << ego->n_sims << ", took " << t3 << endl;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //  ego->n_sims = 1000;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //}

  //for(int i = 2; i < 4; i++) {
  //  ego = reset_ego();
  //  ego->num_points = ego->max_points;
  //  ego->n_sims = 10;
  //  //ego->n_sims = i * 10 / 2;
  //  ego->use_brute_search = true;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  vector<double> x = ego->max_ei_par(i);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute swarm search l=" << i << " nsims " << ego->n_sims << ", took " << t3 << endl;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //  ego->n_sims = 1000;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //}



  //for(int i = 0; i < 1000; i++) {
  //  cout << "Iteration " << i << endl;
  //  ego = reset_ego();
  //  delete ego;
  //}


  //for(int i = 2; i < 6; i++) {
  //  for(int j = 0; j < 4; j++) {
  //  ego = reset_ego();
  //  ego->num_lambda = i;
  //  ego->use_brute_search = true;
  //  //for(int i = -13; i < 14; i++) {
  //  //  for(int j = -13; j < 14; j++) {
  //  //    vector<double> x(2, 0);
  //  //    x[0] = i;
  //  //    x[1] = j;
  //  //    pair<double, double> p = ego->sg->predict(&x[0]);
  //  //    cout << i<< "," << j << " " << p.first << " " << p.second << " "<< ego->ei(p.first, p.second, ego->best_fitness) << endl;
  //  //  }
  //  //}
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  //ego->brute_search_local_swarm(ego->best_particle, i, i, true);
  //  ego->run();
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute Swarm search l=" << i << " num " << j << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
  //  delete ego;
  //  }
  //}
  for(int i = 3; i < 4; i++) {
    ego = reset_ego();
    ego->num_lambda = 3;
    ego->use_brute_search = true;
    ego->suppress = false;
    auto t1 = std::chrono::high_resolution_clock::now();
    ego->max_ei_par(i);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "Brute search l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
    delete ego;

 
    ego = reset_ego();
    ego->num_lambda = 3;
    ego->use_brute_search = false;
    ego->pso_gen = 20;
    ego->suppress = false;
    t1 = std::chrono::high_resolution_clock::now();
    ego->max_ei_par(i);
    t2 = std::chrono::high_resolution_clock::now();
    t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "PSO search l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
  }

  //for(int i = 2; i < 6; i++) {
  //  for(int j = 0; j < 4; j++) {
  //  ego = reset_ego();
  //  ego->num_lambda = i;
  //  ego->use_brute_search = false;
  //  ego->pso_gen = 20;
  //  ego->suppress = false;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  ego->run();
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "PSO search l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
  //  delete ego;
  //  }
  //}
}
