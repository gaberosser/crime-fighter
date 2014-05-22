/*
 * mvn.h
 *
 *  Created on: 16 May 2014
 *      Author: gabriel
 */

#ifndef MVN_H_
#define MVN_H_
#include <string>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
const double PI = 3.141592653589793238463;

class Mvn {
	int ndim;
	vector<double> means;
	vector<double> stdevs;
public:
	Mvn(vector<double> means, vector<double> stdevs) {
		this->ndim = means.size();
		this->means = means;
		this->stdevs = stdevs;
	}
	double pdf(vector<double> x) {
		double a = 1.0;
		double b = 0.0;
		for (int i=0; i<ndim; ++i) {
			a *= stdevs[i];
			b -= (x[i] - means[i]) * (x[i] - means[i]) / (2 * stdevs[i] * stdevs[i]);
		}
		double res = pow(2 * PI, -ndim * 0.5) / a * exp(b);
		return res;
	}
};



#endif /* MVN_H_ */
