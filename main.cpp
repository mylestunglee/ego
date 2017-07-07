#include "ego.hpp"
#include "surrogate.hpp"
#include "evaluator.hpp"
#include "transferrer.hpp"
#include "functions.hpp"
#include "ihs.hpp"

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

	boundaries_t boundaries = {make_pair(-2, 2), make_pair(-2, 2)};

	Evaluator evaluator("./test_script");

	EGO ego(boundaries, evaluator);

	ego.sample_latin(10);
	// Randomness can offset optimiser negatively
	ego.sample_uniform(10);
	ego.run();

	evaluator.save("fitness.log");

	return 0;
}
