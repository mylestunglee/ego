#include <stdexcept>
#include "main.hpp"
#include "ego.hpp"
#include "evaluator.hpp"
#include "transferrer.hpp"
#include "functions.hpp"
#include "csv.hpp"

#include "surrogate.hpp"
#include <utility>

#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
	if (argc != 4 && argc != 5) {
		print_help();
		return 0;
	}

	bool is_knowledge_transfer = argc == 5;

	// Read filenames from arguments
	string filename_script(argv[1]);
	string filename_config(argv[2]);
	string filename_output(argv[3]);

	// Load configuration file
	auto config = read(filename_config);

	size_t max_evaluations, max_trials, constraints, costs;
	double convergence_threshold, sig_level = 1.0, fitness_percentile = 1.0;
	bool is_discrete;
	boundaries_t boundaries;

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
		if (is_knowledge_transfer) {
			sig_level = stof(config.at(8).at(0));
			if (sig_level < 0.0 || sig_level > 1.0) {
				throw invalid_argument("invalid significance level");
			}
			fitness_percentile = stof(config.at(9).at(0));
			if (fitness_percentile < 0.0 || fitness_percentile > 1.0) {
				throw invalid_argument("invalid fitness percentile");
			}
		}
	} catch (const invalid_argument& ia) {
		cerr << "Invalid value in configuration file: " << ia.what() << endl;
		return 0;
	} catch (const out_of_range& oor) {
		cerr << "Insufficent values in configuration file: " << oor.what() << endl;
		return 0;
	}

	// Execute EGO
	Evaluator evaluator(filename_script);

	if (is_knowledge_transfer) {
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
	} else {
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
	}

	evaluator.save(filename_output);
	write_fitness_log("fitnesses.csv");

	return 0;
}

void print_help() {
	cout << "Usage: ego script config log [results]" << endl;
	cout << "\tTo use knowledge transfer, specify some results in CSV format." << endl;
}
