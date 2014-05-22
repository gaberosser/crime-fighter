/*
 * main.cpp
 *
 *  Created on: 16 May 2014
 *      Author: gabriel
 */

#include "mvn.h"
#include "vbkde.h"
#include "libalg/src/alglibmisc.h"
using namespace std;
using namespace alglib;

int main() {
	double m[] = {0.0, 0.0, 0.0};
	double s[] = {1.0, 1.0, 1.0};
	vector<double> means(m, m + sizeof(m)/sizeof(double));
	vector<double> stdevs(s, s + sizeof(s)/sizeof(double));
	Mvn mvn = Mvn(means, stdevs);
	vector<double> x(3, 0.0);

	cout << mvn.pdf(x) << endl;
	cout << "***\n";

	vector<vector<double> > data(1, means);
	FixedBandwidthKde fk(data, stdevs, true);

	double p = fk.pdf(data[0]);
	cout << "Length data: " << data.size() << endl;
	cout << p << endl;

	VariableBandwidthKde vk(data, true);

	real_2d_array a = "[[0,0],[0,1],[1,0],[1,1]]";
    ae_int_t nx = 2;
    ae_int_t ny = 0;
    ae_int_t normtype = 2;
    kdtree kdt;
    real_1d_array x2;
    real_2d_array r = "[[]]";
    real_1d_array z;
    ae_int_t k;
    kdtreebuild(a, nx, ny, normtype, kdt);
    x2 = "[-1,0]";
    k = kdtreequeryknn(kdt, x2, 5);
    printf("%d\n", int(k)); // EXPECTED: 1
    kdtreequeryresultsdistances(kdt, z);
    printf("%s\n", z.tostring(1).c_str()); // EXPECTED: [[0,0]]

	return 0;
}


