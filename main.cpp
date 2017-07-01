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

	vector<pair<double, double>> boundaries = {make_pair(-2, 2), make_pair(-2, 2)};

	Evaluator evaluator("./test_script");

	EGO ego(boundaries, evaluator);

	ego.sample_plan(5 * 2, 5); // 5 * dimension
	ego.run_quad();

	evaluator.save("fitness.log");

	return 0;
}
