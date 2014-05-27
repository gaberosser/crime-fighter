/*
 * vbkde.h
 *
 *  Created on: 16 May 2014
 *      Author: gabriel
 */

#ifndef VBKDE_H_
#define VBKDE_H_
#include "string.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <vector>
#include <exception>

#include "mvn.h"
#include "libalg/src/alglibmisc.h"
using namespace alglib;

class FixedBandwidthKde {
public:
	vector<vector <double> > data;
	vector<vector <double> > bandwidths;
	vector<Mvn> mvns;
	int ndim;
	bool normed;
	FixedBandwidthKde(vector<vector <double> > data, bool normed);
	FixedBandwidthKde(vector<vector <double> > data, vector <double> bdwidths, bool normed);
	FixedBandwidthKde(vector<vector <double> > data, double bdwidth, bool normed);
	void set_mvns(); // TODO move to private
	int ndata() { return data.size(); }
	double pdf(vector <double> x);
	vector<double> pdf(vector< vector<double> > X);
private:
	void set_bandwidths(double bdwidth);
	void set_bandwidths(vector<double> bdwidths);
};

class VariableBandwidthKde: public FixedBandwidthKde {
public:
	VariableBandwidthKde(vector<vector <double> > data, bool normed);
	VariableBandwidthKde(vector<vector <double> > data, int nn, bool normed);
private:
	void set_bandwidths(int nn);
};



#endif /* VBKDE_H_ */
