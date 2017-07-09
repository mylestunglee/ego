#include "ego.hpp"
#include "surrogate.hpp"
#include "evaluator.hpp"
#include "transferrer.hpp"
#include "functions.hpp"
#include "csv.hpp"

using namespace std;

int main(int argc, char* argv[]) {
	double sig_level = 0.05;

	if (argc == 3) {
		string filename_results_old(argv[1]);
		string filename_script_new(argv[2]);
		Evaluator evaluator(filename_script_new);
		Transferrer transferrer(filename_results_old, evaluator, sig_level,
			{make_pair(-3, -1), make_pair(-3, -1)});
		transferrer.run();
		return 0;
	} else {
		string script(argv[1]);
		string filename_config(argv[2]);
		string filename_output(argv[3]);

		auto config = read(filename_config);

		size_t max_evaluations = stoi(config[0][0]);
		size_t max_trials = stoi(config[1][0]);
		double convergence_threshold = stof(config[2][0]);
		boundaries_t boundaries = read_boundaries(config[3], config[4]);

		Evaluator evaluator(script);

		EGO ego(evaluator, boundaries, {}, max_evaluations, max_trials,
			convergence_threshold);

		// Heuristic sample size = 5 * dim
		ego.sample_latin(10);
		// Randomness can offset optimiser negatively
		//ego.sample_uniform(10);
		ego.run();

		evaluator.save(filename_output);

		return 0;
	}
}
