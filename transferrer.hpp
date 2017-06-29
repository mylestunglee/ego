#include <string>
#include <vector>
#include "evaluator.hpp"

using namespace std;

class Transferrer {
	public:
		Transferrer(string filename_results_old, string filename_script_new);
		~Transferrer();
		void transfer();
	private:
		Evaluator* evaluator;
		vector<pair<vector<double>, vector<double>>> old_results;
		vector<pair<double, double>> boundaries;

		void read_results(string filename);
		void calc_correlation(vector<double> x, vector<double> y, double &pearson, double &spearman);
		bool is_bound(vector<double> x);
		vector<pair<vector<double>, vector<double>>> sample_results_old();
};
