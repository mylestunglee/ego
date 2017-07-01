#include "ego.hpp"
#include "surrogate.hpp"
#include "evaluator.hpp"
#include "transferrer.hpp"

using namespace std;

int main(int argc, char* argv[]) {
	double sig_level = 0.05;

	if (argc == 3) {
		string filename_results_old(argv[1]);
		string filename_script_new(argv[2]);
		Transferrer transferrer(filename_results_old, filename_script_new, sig_level);
		transferrer.transfer();
		return 0;
	}

	vector<double> lower = {-2, -2};
	vector<double> upper = {2, 2};
	int dimension = 2;

	Evaluator* evaluator = new Evaluator("./test_script");

	EGO ego(dimension, lower, upper);
	ego.evaluator = evaluator;
	ego.search_type = 2; // PSO
	ego.use_cost = 1; //EI/cost

	ego.sample_plan(5 * dimension, 5);
	ego.run_quad();

	evaluator->save("fitness.log");

	delete evaluator;

	return 0;
}
