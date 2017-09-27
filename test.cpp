#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <vector>
#include <time.h>
#include <set>
#include "functions.hpp"
#include "compare.hpp"

using namespace std;

TEST_CASE("is_bounded") {
	REQUIRE(is_bounded({2.0}, {make_pair(1.0, 3.0)}));
	REQUIRE(!is_bounded({0.0}, {make_pair(1.0, 3.0)}));
	REQUIRE(is_bounded({1.0}, {make_pair(1.0, 3.0)}));
	REQUIRE(!is_bounded({1.0, 2.0},
		{make_pair(1.0, 3.0), make_pair(0.0, 1.0)}));
}

TEST_CASE("generate_uniform_space") {
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng, time(NULL));
	boundaries_t boundaries = {make_pair(0.0, 1.0)};
	auto point = generate_uniform_sample(rng, boundaries);
	REQUIRE(is_bounded(point, boundaries));
	gsl_rng_free(rng);
}

TEST_CASE("euclidean_distance") {
	REQUIRE(euclidean_distance({2.0}, {3.0}) == 1.0);
	REQUIRE(euclidean_distance({0.0, 3.0}, {4.0, 0.0}) == 5.0);
}

TEST_CASE("success_probabilty") {
	REQUIRE(success_probability(0.0, 0.0) == 0.0);
	REQUIRE(success_probability(0.5, 1.0) >  0.0);
	REQUIRE(success_probability(1.0, 0.0) == 1.0);
}

TEST_CASE("get_intersection") {
	boundaries_t b1 = {make_pair(0.0, 2.0)};
	boundaries_t b2 = {make_pair(1.0, 3.0)};
	boundaries_t b3 = {make_pair(1.0, 2.0)};
	REQUIRE(get_intersection(b1, b2) == b3);

	boundaries_t b4 = {make_pair(3.0, 4.0)};
	REQUIRE(get_intersection(b3, b4).empty());

	boundaries_t b5 = {make_pair(2.0, 3.0)};
	REQUIRE(get_intersection(b3, b4).empty());
}

TEST_CASE("generate_latin_samples") {
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus);
	boundaries_t boundaries = {make_pair(0.0, 10.0), make_pair(0.0, 1.0)};
	auto points = generate_latin_samples(rng, 10, boundaries);
	set<double> xs;
	set<double> ys;
	for (auto point : points) {
		REQUIRE(is_bounded(point, boundaries));
		auto x = point[0];
		auto y = point[1];
		REQUIRE(xs.find(x) == xs.end());
		REQUIRE(ys.find(y) == ys.end());
		xs.insert(x);
		ys.insert(y);
	}
	gsl_rng_free(rng);
}

TEST_CASE("infer_boundaries") {
	boundaries_t boundaries = {make_pair(1.0, 3.0)};
	results_t r1 = {
		make_pair(vector<double>{1.0}, vector<double>{}),
		make_pair(vector<double>{2.0}, vector<double>{}),
		make_pair(vector<double>{3.0}, vector<double>{})};
	REQUIRE(infer_boundaries(r1) == boundaries);

	results_t r2 = {
		make_pair(vector<double>{1.0}, vector<double>{}),
		make_pair(vector<double>{1.0}, vector<double>{})};
	REQUIRE(infer_boundaries(r2).empty());
}

TEST_CASE("apply_polynomial") {
	REQUIRE(apply_polynomial(1.0, {1.0, 1.0}) == 2.0);
	REQUIRE(apply_polynomial(1.0, {}) == 0.0);
	REQUIRE(apply_polynomial(3.0, {4.0, 3.0, 1.0}) == 22.0);
}

TEST_CASE("is_subset") {
	boundaries_t b1 = {make_pair(1.0, 3.0)};
	boundaries_t b2 = {make_pair(3.0, 4.0)};
	boundaries_t b3 = {make_pair(1.0, 5.0)};
	REQUIRE(is_subset(b1, b3));
	REQUIRE(!is_subset(b1, b2));
	REQUIRE(is_subset(b2, b2));
}

TEST_CASE("calc_correlation") {
	double pearson;
	double spearman;
	calc_correlation({1.0, 2.0, 3.0}, {1.0, 2.3, 1.6}, pearson, spearman);
	REQUIRE(abs(pearson - 0.461) < 0.01);
	REQUIRE(abs(spearman - 0.5) < 0.01);
}

TEST_CASE("are_valid_boundaries") {
	boundaries_t b1 = {make_pair(1.0, 3.0)};
	boundaries_t b2 = {make_pair(1.0, 1.0)};
	boundaries_t b3 = {make_pair(9.0, 1.0)};
	REQUIRE(are_valid_boundaries(b1));
	REQUIRE(!are_valid_boundaries(b2));
	REQUIRE(!are_valid_boundaries(b3));
}

TEST_CASE("calc_hypervolume") {
	boundaries_t b1 = {make_pair(1.0, 3.0)};
	boundaries_t b2 = {make_pair(-1.0, 1.0), make_pair(-1.0, 1.0)};
	REQUIRE(calc_hypervolume(b1) == 2.0);
	REQUIRE(calc_hypervolume(b2) == 4.0);
}

TEST_CASE("read_boundaries") {
	boundaries_t b1 = {make_pair(1.0, 3.0)};
	boundaries_t b2 = {make_pair(2.0, 4.0), make_pair(3.0, 5.0)};
	REQUIRE(read_boundaries({"1.0"}, {"3.0"}) == b1);
	REQUIRE(read_boundaries({"2.0", "3.0"}, {"4.0", "5.0"}) == b2);
}

TEST_CASE("round_vector") {
	vector<double> x1 = {1.2, 2.5, -0.2};
	vector<double> x2 = {1.0, 3.0, 0.0};
	vector<double> x3 = {1.0, 2.0, -1.0};
	REQUIRE(round_vector(x1) == x2);
	REQUIRE(round_vector(x3) == x3);
}

TEST_CASE("generate_grid_samples") {
	boundaries_t boundaries = {make_pair(0.0, 1.0), make_pair(0.0, 1.0)};
	vector<vector<double>> points = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
	REQUIRE(generate_grid_samples(2, boundaries) == points);
}

static double quadratic(const gsl_vector* v, void*) {
	auto x = gsl_to_std_vector(v);
	return x[0] * x[0];
}

static vector<double> generator(void*) {
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus);
	auto result = {gsl_rng_uniform(rng) - 0.5};
	gsl_rng_free(rng);
	return result;
}

TEST_CASE("minimise_local") {
	double minimum = numeric_limits<double>::max();
	auto x = minimise_local(quadratic, NULL, generator(NULL), 0.0, 1000, minimum);
	REQUIRE(abs(x[0]) < 0.001);
	REQUIRE(minimum < 0.001);
}

TEST_CASE("minimise") {
	double minimum = numeric_limits<double>::max();
	auto x = minimise(quadratic, generator, NULL, 0.0, 1000,
		[](auto){return true;}, minimum);
	REQUIRE(abs(x[0]) < 0.001);
	REQUIRE(minimum < 0.001);
}

TEST_CASE("gsl_to_std_vector") {
	gsl_vector* v = gsl_vector_alloc(2);
	gsl_vector_set(v, 0, 0.0);
	gsl_vector_set(v, 1, 1.0);
	vector<double> x = {0.0, 1.0};
	REQUIRE(gsl_to_std_vector(v) == x);
	gsl_vector_free(v);
}

TEST_CASE("generate_all_samples") {
	boundaries_t b1 = {make_pair(0.0, 2.0), make_pair(0.0, 2.0)};
	auto ps1 = generate_grid_samples(3, b1);
	auto ps2 = generate_all_samples(b1);
	sort(ps1.begin(), ps1.end());
	sort(ps2.begin(), ps2.end());
	REQUIRE(ps1 == ps2);
}

TEST_CASE("calc_midpoint") {
	vector<vector<double>> ps1 = {{1.0}, {2.0}};
	vector<double> m1 = {1.5};
	vector<vector<double>> ps2 = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}};
	vector<double> m2 = {2.0, 3.0};
	REQUIRE(calc_midpoint(ps1) == m1);
	REQUIRE(calc_midpoint(ps2) == m2);
}

TEST_CASE("multilinear_regression_fit") {
	vector<vector<double>> xs = {{1.0}, {2.0}, {3.0}};
	vector<double> ys = {2.0, 4.0, 6.0};
	vector<double> coeffs = {2.0};
	REQUIRE(multilinear_regression_fit(xs, ys) == coeffs);
}

TEST_CASE("multquadratic_regression_fit") {
	vector<vector<double>> xs = {{1.0}, {2.0}, {3.0}};
	vector<double> ys = {1.0, 4.0, 9.0};
	vector<double> coeffs = {0.0, 0.0, 1.0};
}

TEST_CASE("multiquadratic_result_extrapolate") {
	results_t results = {
		make_pair(vector<double>{1.0}, vector<double>{1.0, 0.0}),
		make_pair(vector<double>{2.0}, vector<double>{4.0, 0.0}),
		make_pair(vector<double>{3.0}, vector<double>{9.0, 0.0}),
		make_pair(vector<double>{4.0}, vector<double>{0.0, 1.0})};

	vector<double> coeffs = {0.0, 0.0, 1.0};
//	REQUIRE(euclidean_distance(
//		multiquadratic_result_extrapolate(results, 0, 0)[0], coeffs) < 0.001);
}

TEST_CASE("minimise_multiquadratic") {
	vector<vector<double>> fs1 = {{0.0, 0.0, 1.0}};
	vector<vector<double>> fs2 = {{4.0, 4.0, 1.0}};
	vector<vector<double>> fs3 = {{0.0, 0.0, -1.0}, {0.0, 0.0, -1.0}};

	vector<double> p1 = {0.0};
	vector<double> p2 = {-1.0};
	vector<double> p3 = {1.0, 1.0};

	boundaries_t b1 = {make_pair(-1.0, 1.0)};
	boundaries_t b2 = {make_pair(0.0, 1.0), make_pair(0.0, 1.0)};
	REQUIRE(minimise_multiquadratic(fs1, b1) == p1);
	REQUIRE(minimise_multiquadratic(fs2, b1) == p2);
	REQUIRE(minimise_multiquadratic(fs3, b2) == p3);
}

TEST_CASE("prune_boundaries") {

}

TEST_CASE("sample_mean") {
	REQUIRE(sample_mean({1.0, 2.0, 3.0}) == 2.0);
}

TEST_CASE("sample_sd") {
	REQUIRE(sample_sd({1.0, 2.0, 3.0}) == 1.0);
}

TEST_CASE("log_vector") {
	REQUIRE(log_vector({2.0})[0] == log(2.0));
	REQUIRE(log_vector({}).empty());
}

TEST_CASE("count_common_results") {
	results_t r1 = {
		make_pair(vector<double>{1.0}, vector<double>{}),
		make_pair(vector<double>{2.0}, vector<double>{}),
		make_pair(vector<double>{3.0}, vector<double>{})};
	results_t r2 = {
		make_pair(vector<double>{3.0}, vector<double>{})};
	results_t r3 = {
		make_pair(vector<double>{2.0}, vector<double>{}),
		make_pair(vector<double>{3.0}, vector<double>{})};
	REQUIRE(count_common_results({r1}, r2) == 1);
	REQUIRE(count_common_results({r2, r3}) == 1);
	REQUIRE(count_common_results({r3, r1}) == 2);
}

TEST_CASE("extract_cluster_midpoint") {
	results_t r1 = {
		make_pair(vector<double>{1.0}, vector<double>{}),
		make_pair(vector<double>{3.0}, vector<double>{})};
	results_t r2 = {
		make_pair(vector<double>{1.0}, vector<double>{}),
		make_pair(vector<double>{4.0}, vector<double>{})};
	vector<results_t> rs = {r1, r2};
	vector<double> p1 = {1.0};
	vector<double> p2 = {3.5};
	REQUIRE(extract_cluster_midpoint(rs) == p1);
	REQUIRE(extract_cluster_midpoint(rs) == p2);
	REQUIRE(rs[0].empty());
	REQUIRE(rs[1].empty());
}

TEST_CASE("are_unique_names") {
	REQUIRE(are_unique_names({"alice", "bob", "charlie"}));
	REQUIRE(!are_unique_names({"alice", "bob", "alice"}));
}

TEST_CASE("swap_pattern_coeffss") {
	vector<coeffs_t> c1 = {{1.0}, {2.0}, {3.0}};
	vector<coeffs_t> c2 = {{1.0}, {3.0}, {2.0}};
	vector<string> n1 = {"x", "y", "z"};
	vector<string> n2 = {"x", "z", "y"};

	REQUIRE(swap_pattern_coeffss(n1, n2, c1) == c2);
}
