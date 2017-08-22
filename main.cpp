#include <stdexcept>
#include "main.hpp"
#include "ego.hpp"
#include "evaluator.hpp"
#include "transferrer.hpp"
#include "functions.hpp"
#include "csv.hpp"
#include "compare.hpp"

using namespace std;

int main(int argc, char* argv[]) {
	// Require at least one argument
	if (argc < 2) {
		cerr << "Insufficent arguments" << endl;
		print_help(cerr);
		return 1;
	}

	// Determine mode
	string mode_str(argv[1]);
	Mode mode = Mode::undefined;
	if (mode_str == "-o" || mode_str == "--optimise") {
		mode = Mode::optimise;
	} else if (mode_str == "-t" || mode_str == "--transfer") {
		mode = Mode::transfer;
	} else if (mode_str == "-c" || mode_str == "--compare") {
		mode = Mode::compare;
	} else if (mode_str == "-h" || mode_str == "-?" || mode_str == "--help") {
		print_help(cout);
		return 0;
	} else {
		cerr << "Invalid mode" << endl;
		return 1;
	}

	// Check number of arguments
	if ((mode == Mode::compare && argc < 4) ||
		(mode == Mode::optimise && (argc < 5 || argc > 6)) ||
		(mode == Mode::transfer && (argc < 6 || argc % 2 != 0))) {
		cerr << "Invalid number of arguments" << endl;
		return 1;
	}

	if (mode == Mode::compare) {
		// Read new results and configs
		string filename_config_new(argv[2]);
		config_t config_new;
		bool error = read_config(filename_config_new, config_new);

		if (error) {
			return 1;
		}

		// Read old results and configs
		string filename_results_new(argv[3]);
		results_t results_new = read_results(
			filename_results_new, config_new.boundaries.size());
		vector<results_t> results_olds;
		vector<config_t> configs_old;
		for (int i = 0; i < (argc - 4) / 2; i++) {
			// Read config
			string filename_config_old(argv[4 + 2 * i]);
			config_t config_old;
			bool error = read_config(filename_config_old, config_old);
			if (error) {
				return 1;
			}
			configs_old.push_back(config_old);

			// Read result
			string filename_results_old(argv[5 + 2 * i]);
			results_olds.push_back(
				read_results(filename_results_old, config_old.boundaries.size()));
		}

		compare(config_new, results_new, configs_old, results_olds, argv);

		return 0;
	}


	// Read filenames from arguments
	string filename_script(argv[2]);
	string filename_config(argv[3]);
	string filename_output(argv[4]);

	// Load evaluator
	Evaluator evaluator(filename_script);

	// Load configuration file
	config_t config;
	bool error = read_config(filename_config, config);
	size_t dimension = config.boundaries.size();

	if (error) {
		return 1;
	}

	if (mode == Mode::optimise) {
		boundaries_t rejection = {};
		results_t results = {};

		// Read sampled results if provided
		if (argc == 6) {
			string filename_results(argv[5]);
			results = read_results(filename_results, dimension);
			rejection = infer_boundaries(results);
		}

		EGO ego(evaluator, config, rejection);

		if (argc == 6) {
			simulate_results(ego, results);
		}

		// Heuristic sample size = 5 * dim
		cout << "Sampling using LHS" << endl;
		ego.sample_latin(5 * dimension);
		// Randomness can affect optimiser negatively
		cout << "Sampling using uniform" << endl;
		ego.sample_uniform(5 * dimension);
		cout << "Running EGO" << endl;
		ego.run();
	} else {
		string filename_results_old(argv[5]);
		results_t results_old = read_results(filename_results_old, dimension);
		results_t results_new;

		if (argc == 7) {
			string filename_results_new(argv[6]);
			results_new = read_results(filename_results_new, dimension);
		}

		Transferrer transferrer(results_old, results_new, evaluator, config);
		transferrer.run();
	}

	evaluator.save(filename_output);
	write_fitness_log("fitnesses.csv");

	return 0;
}

// Prints usage information
void print_help(ostream& cstr) {
	cstr << "Usage:" << endl;
	cstr << "\tego -o script config output [results]" << endl;
	cstr << "\tego -t script config output results_old [results_new]" << endl;
	cstr << "\tego -c config_new results_new configs_old... results_old..." << endl;
	cstr << "\tego -h" << endl;
	cstr << "\t-o --optimise Optimise" << endl;
	cstr << "\t-t --transfer Transfer" << endl;
	cstr << "\t-c --compare  Compare" << endl;
}

// Given an optimiser, simulate sampling using provided results
void simulate_results(EGO& ego, results_t results) {
	for (auto result : results) {
		ego.simulate(result.first, result.second);
	}
}
