#include <iostream>
#include "compare.hpp"

using namespace std;

void compare(results_t& results_new, vector<results_t>& results_olds) {
	cout << "comparing!" << endl;
	cout << covariance_results(results_new) << endl;
	cout << results_olds.size() << endl;
}
