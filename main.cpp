#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {

  double (*fitness)(double x[]) = &easy_test;
  int dimension = 2;
  vector<double> lower = {0.0, 0.0};
  vector<double> upper = {10.0, 10.0};
  int which = 1;

  if(argc >= 2) {
    which = atoi(argv[1]);
    switch(which) {
      case 1:
        fitness = &easy_test;
	dimension = 2;
        break;
      case 2:
        fitness = &ackley;
	dimension = 2;
        lower = {-5.0, -5.0};
        upper = {5.0, 5.0};
        break;
      case 3:
        fitness = &goldstein;
	dimension = 2;
        lower = {-2.0, -2.0};
        upper = {2.0, 2.0};
        break;
      case 4:
        fitness = &sphere_4;
	dimension = 4;
        lower = {-10, -10, -10, -10};
        upper = {10, 10, 10, 10};
        break;
      case 5:
        fitness = &sphere_3;
	dimension = 3;
        lower = {-10, -10, -10};
        upper = {10, 10, 10};
        break;
    }
  }
    
  Surrogate sg(dimension, SEiso);
  sg.set_params(0.0, 0.0);
  EGO ego(dimension, &sg, lower, upper, fitness);

  if(which == 1) {
    for(double i = 1; i < 3; ++i) {
      for(double z = 1; z < 3; ++z) {
        vector<double> x = {i * 5, z*5};
        double y = fitness(&x[0]);
        ego.add_training(x, y);
      }
    }
    ego.max_fitness = -33.75;
    ego.max_iterations = 50;
    ego.is_discrete = true;
  } else if(which == 2) {
    for(double i = -5.0; i <= 5; i += 2) {
      for(double z = -5.0; z <= 5; z += 2) {
        vector<double> x = {i, z};
        double y = fitness(&x[0]);
        ego.add_training(x, y);
      }
    }
    ego.max_fitness = 0;
    ego.max_iterations = 50;
    //ego.is_discrete = true;
  } else if(which == 3) {
    for(double i = -2.0; i <= 2; i += 4) {
      for(double z = -2.0; z <= 2; z += 4) {
        vector<double> x = {i, z};
        double y = fitness(&x[0]);
        ego.add_training(x, y);
      }
    }
    ego.max_fitness = 3;
    ego.max_iterations = 50;
    ego.is_discrete = true;
  } else if(which == 4) {
    for(double i = -10; i <= 10; i += 5) {
      for(double j = -10; j <= 10; j += 5) {
        for(double k = -10; k <= 10; k += 5) {
          for(double z = -10; z <= 10; z += 5) {
	    if(i != 0 || j != 0 || k != 0 || z != 0) {
              vector<double> x = {i, j, k, z};
              double y = fitness(&x[0]);
              ego.add_training(x, y);
	    }
	  }
	}
      }
    }
    ego.max_fitness = 0;
    ego.max_iterations = 1000;
    ego.is_discrete = true;
    ego.use_brute_search = true;
    ego.num_lambda = 2;
    ego.num_points = 4;
    ego.n_sims = 20;
  } else if(which == 5) {
    for(double i = -10; i <= 10; i += 5) {
      for(double j = -10; j <= 10; j += 5) {
        for(double k = -10; k <= 10; k += 5) {
	  if(i != 0 || j != 0 || k != 0) {
            vector<double> x = {i, j, k};
            double y = fitness(&x[0]);
            ego.add_training(x, y);
	  }
	}
      }
    }
    ego.max_fitness = 0;
    ego.max_iterations = 1000;
    ego.min_expected_imp = 10;
    ego.is_discrete = true;
    ego.n_sims = 500;
    ego.population_size = 1;
    ego.num_lambda = 2;
    ego.num_points = 5;
    //ego.use_brute_search = true;
  }

  ego.run();

  cout << "BEST: " << endl;
  for(int i = 0; i < ego.dimension; i++) {
    cout << ego.best_result()[i] << " ";
  }
  cout << ": best result " << ego.best_fitness << endl;
}

