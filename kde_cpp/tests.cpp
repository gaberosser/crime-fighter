/*
 * tests.cpp
 *
 *  Created on: 21 May 2014
 *      Author: gabriel
 */
#include "cpptest.h"
#include "mvn.h"
#include "vbkde.h"
#include <iostream>
#include <math.h>

using namespace std;
const double eps = 1e-12;

class KdeTestSuite : public Test::Suite
{
public:
	KdeTestSuite()
    {
        TEST_ADD(KdeTestSuite::test_mvn);
        TEST_ADD(KdeTestSuite::test_fixed_bandwidth_kde);
        TEST_ADD(KdeTestSuite::test_variable_bandwidth_kde);
    }

private:
    void test_mvn();
    void test_fixed_bandwidth_kde();
    void test_variable_bandwidth_kde();
};

void KdeTestSuite::test_mvn()
{
	double m[] = {0.0, 0.0, 0.0};
	double s[] = {1.0, 2.0, 3.0};
	vector<double> means(m, m + sizeof(m)/sizeof(double));
	vector<double> stdevs(s, s + sizeof(s)/sizeof(double));
	Mvn mvn = Mvn(means, stdevs);
	vector<double> x(3, 0.0);
	TEST_ASSERT_DELTA(mvn.pdf(x), pow(2*PI, -1.5) / 6.0, eps);

	x[0] = 1.0;
	x[1] = 1.0;
	x[2] = 1.0;
	double a = pow(2*PI, -1.5) / 6.0;
	double b = exp(-1.0 / 2.0 - 1.0 / 8.0 - 1.0 / 18.0);
	TEST_ASSERT_DELTA(mvn.pdf(x), a * b, eps);

	double m2[] = {1.0, 0.0, 0.0};
	vector<double> means2(m2, m2 + sizeof(m2)/sizeof(double));
	mvn = Mvn(means2, stdevs);
	b = exp(-1.0 / 8.0 - 1.0 / 18.0);
	TEST_ASSERT_DELTA(mvn.pdf(x), a * b, eps);
}

void KdeTestSuite::test_fixed_bandwidth_kde()
{
	double m[] = {0.0, 0.0, 0.0};
	double s[] = {1.0, 2.0, 3.0};
	vector<double> means(m, m + sizeof(m)/sizeof(double));
	vector<double> stdevs(s, s + sizeof(s)/sizeof(double));
	double x1_[] = {0.0, 0.0, 0.0};
	double x2_[] = {1.0, 0.0, 0.0};
	double x3_[] = {0.5, -1.0, 0.0};
	vector<double> x1(x1_, x1_ + 3);
	vector<double> x2(x2_, x2_ + 3);
	vector<double> x3(x3_, x3_ + 3);
	vector<vector<double> > data;
	data.push_back(x1);
	data.push_back(x2);
	// normed KDE
	FixedBandwidthKde fk(data, stdevs, true);
	TEST_ASSERT(int(fk.mvns.size()) == fk.ndata());

	double p1 = fk.pdf(x1);
	double p2 = fk.pdf(x3);
	double expct1 = pow(2*PI, -1.5) / 6.0 * (1.0 + exp(-0.5));
	double expct2 = pow(2*PI, -1.5) / 6.0 * (exp(-0.25));
	TEST_ASSERT_DELTA(p1, expct1 * 0.5, eps);
	TEST_ASSERT_DELTA(p2, expct2, eps);
	// unnormed KDE
	fk.normed = false;
	p1 = fk.pdf(x1);
	p2 = fk.pdf(x3);
	TEST_ASSERT_DELTA(p1, expct1, eps);
	TEST_ASSERT_DELTA(p2, expct2 * 2.0, eps);

	// pdf call with multiple targets
	vector<vector<double> > targets;
	targets.push_back(x1);
	targets.push_back(x2);
	targets.push_back(x3);
	vector <double> P = fk.pdf(targets);
	TEST_ASSERT_DELTA(P[0], expct1, eps);
	TEST_ASSERT_DELTA(P[1], expct1, eps);
	TEST_ASSERT_DELTA(P[2], expct2 * 2.0, eps);

}

void KdeTestSuite::test_variable_bandwidth_kde() {

}


int main() {
	KdeTestSuite ets;
    Test::TextOutput output(Test::TextOutput::Verbose);
    return ets.run(output, false); // Note the 'false' parameter
}
