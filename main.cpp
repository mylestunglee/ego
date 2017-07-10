#include "ego.hpp"
#include "surrogate.hpp"
#include "evaluator.hpp"
#include "transferrer.hpp"
#include "functions.hpp"
#include "csv.hpp"

using namespace std;

int main(int argc, char* argv[]) {
	if (argc == 5) {
		// Using knowledge transfer
		string filename_results_old(argv[1]);
		string filename_script(argv[2]);
		string filename_config(argv[3]);
		string filename_output(argv[4]);

		auto config = read(filename_config);

		size_t max_evaluations = stoi(config[0][0]);
		size_t max_trials = stoi(config[1][0]);
		double convergence_threshold = stof(config[2][0]);
		bool is_discrete = config[3][0].compare("true") == 0;
		size_t constraints = stoul(config[4][0]);
		size_t costs = stoul(config[5][0]);
		boundaries_t boundaries = read_boundaries(config[6], config[7]);
		double sig_level = stof(config[8][0]);

		Evaluator evaluator(filename_script);
		Transferrer transferrer(
			filename_results_old, evaluator, max_evaluations,
			max_trials, convergence_threshold, sig_level, boundaries,
			is_discrete, constraints, costs);
		transferrer.run();
		evaluator.save(filename_output);
		return 0;
	} else if (argc == 4) {
		// Not using knowledge transfer
		string script(argv[1]);
		string filename_config(argv[2]);
		string filename_output(argv[3]);

		auto config = read(filename_config);

		size_t max_evaluations = stoul(config[0][0]);
		size_t max_trials = stoul(config[1][0]);
		double convergence_threshold = stof(config[2][0]);
		// TODO: handle neither true or false exception
		bool is_discrete = config[3][0].compare("true") == 0;
		size_t constraints = stoul(config[4][0]);
		size_t costs = stoul(config[5][0]);
		boundaries_t boundaries = read_boundaries(config[6], config[7]);

		Evaluator evaluator(script);

		EGO ego(evaluator, boundaries, {}, max_evaluations, max_trials,
			convergence_threshold, is_discrete, constraints, costs);

		// Heuristic sample size = 5 * dim
		ego.sample_latin(10);
		// Randomness can offset optimiser negatively
		//ego.sample_uniform(10);
		ego.run();
		evaluator.save(filename_output);


		return 0;
	}

	cout << "Usage:" << endl;
	cout << "Without knowledge transfer: ego script config log" << endl;
	cout << "Script calls the fitness function; config contains hyperparameters for this program; log is the output that stores calls to the script." << endl;
	cout << "With knowledge transfer: ego results script config log" << endl;
	cout << "Results is the log of the previous (optimised) design." << endl;

	return 0;
}
