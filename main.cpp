#include <stdexcept>
#include "main.hpp"
#include "ego.hpp"
#include "evaluator.hpp"
#include "transferrer.hpp"
#include "functions.hpp"
#include "csv.hpp"

using namespace std;

int main(int argc, char* argv[]) {
	// Require at least one argument
	if (argc < 2) {
		cerr << "Insufficent arguments" << endl;
		print_help();
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
	} else if (mode_str == "--help") {
		print_help();
		return 0;
	} else {
		cerr << "Invalid mode" << endl;
		return 1;
	}

	if (mode == Mode::compare) {
		cerr << "Not implemented" << endl;
		return 0;
	}

	if ((mode == Mode::optimise && (argc < 5 || argc > 6)) ||
		(mode == Mode::transfer && (argc < 6 || argc > 7))) {
		cerr << "Invalid number of arguments" << endl;
		return 1;
	}

	// Read filenames from arguments
	string filename_script(argv[2]);
	string filename_config(argv[3]);
	string filename_output(argv[4]);

	// Load evaluator
	Evaluator evaluator(filename_script);

	// Load configuration file
	size_t max_evaluations, max_trials, constraints, costs;
	double convergence_threshold, sig_level, fitness_percentile;
	bool is_discrete;
	boundaries_t boundaries;
	bool error = read_config(filename_config, max_evaluations, max_trials,
		constraints, costs, convergence_threshold, sig_level,
		fitness_percentile, is_discrete, boundaries);

	if (error) {
		return 1;
	}

	if (mode == Mode::optimise) {
		EGO ego(
			evaluator,
			boundaries,
			{},
			max_evaluations,
			max_trials,
			convergence_threshold,
			is_discrete,
			constraints,
			costs);
		// Heuristic sample size = 5 * dim
		cout << "Sampling using LHS" << endl;
		ego.sample_latin(5 * boundaries.size());
		// Randomness can affect optimiser negatively
		cout << "Sampling using uniform" << endl;
		ego.sample_uniform(5 * boundaries.size());
		cout << "Running EGO" << endl;
		ego.run();
		return 0;
	} else {
		string filename_results_old(argv[4]);

		Transferrer transferrer(
			filename_results_old,
			evaluator,
			max_evaluations,
			max_trials,
			convergence_threshold,
			sig_level,
			boundaries,
			is_discrete,
			constraints,
			costs,
			fitness_percentile);
		transferrer.run();
	}

	evaluator.save(filename_output);
	write_fitness_log("fitnesses.csv");

	return 0;
}

// Prints usage information
void print_help() {
	cerr << "Usage:" << endl;
	cerr << "\tego -o script config output [results]" << endl;
	cerr << "\tego -t script config output results_old [results_new]" << endl;
	cerr << "\tego -c config output results_new results_old_1..." << endl;
	cerr << "\t-o\tOptimise" << endl;
	cerr << "\t-t\tTransfer" << endl;
	cerr << "\t-c\tCompare" << endl;
}

// Attempts to read configuration file at filename, returns false iff
// configuration file loads successfully
bool read_config(
    string filename,
    size_t& max_evaluations,
    size_t& max_trials,
    size_t& constraints,
    size_t& costs,
    double& convergence_threshold,
    double& sig_level,
    double& fitness_percentile,
    bool& is_discrete,
    boundaries_t& boundaries) {
	auto config = read(filename);

	// Parse configuration file
	try {
		max_evaluations = stol(config.at(0).at(0));
		max_trials = stol(config.at(1).at(0));
		convergence_threshold = stof(config.at(2).at(0));
		if (convergence_threshold < 0.0) {
			throw invalid_argument("invalid convergence threshold");
		}
		string is_discrete_string = config.at(3).at(0);
		if (is_discrete_string.compare("true") != 0 &&
				is_discrete_string.compare("false") != 0) {
			throw invalid_argument("is_discrete is not a boolean");
		}
		is_discrete = config.at(3).at(0).compare("true") == 0;
		constraints = stoul(config.at(4).at(0));
		costs = stoul(config.at(5).at(0));
		boundaries = read_boundaries(config.at(6), config.at(7));
		if (!are_valid_boundaries(boundaries)) {
			throw invalid_argument("invalid boundaries");
		}
		sig_level = stof(config.at(8).at(0));
		if (sig_level < 0.0 || sig_level > 1.0) {
			throw invalid_argument("invalid significance level");
		}
		fitness_percentile = stof(config.at(9).at(0));
		if (fitness_percentile < 0.0 || fitness_percentile > 1.0) {
			throw invalid_argument("invalid fitness percentile");
		}
	} catch (const invalid_argument& ia) {
		cerr << "Invalid value in configuration file: " << ia.what() << endl;
		return true;
	} catch (const out_of_range& oor) {
		cerr << "Insufficent values in configuration file: " << oor.what() << endl;
		return true;
	}

	return false;
}
