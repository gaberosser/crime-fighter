/*
 * vbkde.cpp
 *
 *  Created on: 16 May 2014
 *      Author: gabriel
 */

#include "vbkde.h"

real_2d_array vector_2d_to_array(vector<vector<double> > x) {
	unsigned int rows = x.size();
	unsigned int cols = x[0].size();
	real_2d_array res;
	res.setlength(rows, cols);
	double arr[rows*cols];
	for (unsigned int i=0; i<rows; ++i) {
		for (unsigned int j=0; j<cols; ++j) {
			arr[j + i*cols] = x[i][j];
		}
	}
	res.setcontent(rows, cols, arr);
	return res;
}

double st_dev(vector<double> v, bool ub=true) {
	double k = ub ? v.size() - 1 : v.size();
	double sum = std::accumulate(v.begin(), v.end(), 0.0);
	double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
	return sqrt((sq_sum - sum * sum / double(v.size())) / k);
}

FixedBandwidthKde::FixedBandwidthKde(vector<vector <double> > data, bool normed) {
	this->ndata = data.size();
	this->ndim = data[0].size();
	this->normed = normed;
	this->data = data;
}

FixedBandwidthKde::FixedBandwidthKde(vector<vector <double> > data, vector<double> bdwidths, bool normed) {
	this->ndata = data.size();
	this->ndim = data[0].size();
	this->normed = normed;
	this->data = data;
	this->set_bandwidths(bdwidths);
	this->set_mvns();
}

FixedBandwidthKde::FixedBandwidthKde(vector<vector <double> > data, double bdwidth, bool normed) {
	this->ndim = data.size();
	this->ndata = data[0].size();
	this->normed = normed;
	this->data = data;
	this->set_bandwidths(bdwidth);
	this->set_mvns();
}

void FixedBandwidthKde::set_bandwidths(double bdwidth) {
	this->bandwidths = vector<vector<double> >(ndata, vector<double>(ndim, bdwidth));
}

void FixedBandwidthKde::set_bandwidths(vector<double> bdwidths) {
	this->bandwidths = vector<vector<double> >(ndata, bdwidths);
}

void FixedBandwidthKde::set_mvns() {
	for (int i=0; i<ndata; ++i) {
		this->mvns.push_back(Mvn(data[i], this->bandwidths[i]));
	}
}

double FixedBandwidthKde::pdf(vector <double> x) {
//	if (x.size() != ndim) {
//		throw exception("Input vector has wrong size");
//	}
	double res = 0;
	for (int i=0; i<ndata; ++i) {
		res += mvns[i].pdf(x);
	}
	return res;
}

VariableBandwidthKde::VariableBandwidthKde(vector<vector <double> > data, bool normed) : FixedBandwidthKde(data, normed) {

}

void VariableBandwidthKde::set_bandwidths(int nn) {
	this->bandwidths = vector<vector<double> >(ndata, vector<double>(ndim, 0.0));
	// normalise data according to stdev
	vector<double> stds(3, 0.0);
	for (int i=0; i<ndim; ++i) {

		vector<double> this_vec;
		for (int j=0; j<ndim; ++j) {
			this_vec.push_back(data[j][i]);
		}
		stds[i] = st_dev(this_vec);
	}

	// get bandwidths using nearest neighbours
	real_2d_array xy = vector_2d_to_array(data);
	kdtree kd;
	kdtreebuild(xy, this->ndim, 0, 2, kd);

	// iterate over each source and find NN distance
	for (int i=0; i<this->ndata; ++i) {
		real_1d_array pt, z;
		pt.setcontent(this->ndim, &(this->data[i][0]));
		kdtreequeryknn(kd, pt, nn, false);
		kdtreequeryresultsdistances(kd, z);

	}
}

