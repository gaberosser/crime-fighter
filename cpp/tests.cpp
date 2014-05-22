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
    }

private:
    void test_mvn();
    void test_fixed_bandwidth_kde();
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
	double x3_[] = {0.5, 0.0, 0.0};
	vector<double> x1(x1_, x1_ + 3);
	vector<double> x2(x2_, x2_ + 3);
	vector<double> x3(x3_, x3_ + 3);
	vector<vector<double> > data;
	data.push_back(x1);
	data.push_back(x2);
	// normed KDE
	FixedBandwidthKde fk(data, stdevs, true);
	double p1 = fk.pdf(x1);
	cout << p1 << endl;
	TEST_ASSERT_DELTA(p1, pow(2*PI, -1.5) / 6.0, eps);
	// unnormed KDE
	fk.normed = false;
	p1 = fk.pdf(data[0]);
	cout << p1 << endl;
	TEST_ASSERT_DELTA(p1, pow(2*PI, -1.5) / 6.0, eps);

}

int main() {
	KdeTestSuite ets;
    Test::TextOutput output(Test::TextOutput::Verbose);
    return ets.run(output, false); // Note the 'false' parameter
}
