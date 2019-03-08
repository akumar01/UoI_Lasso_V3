#ifndef LASSO_H
#define LASSO_H

#include <eigen3/Eigen/Dense>
//#include <boost/tuple/tuple.hpp>
#include <tuple>

using namespace std;
using namespace Eigen; 

#ifdef __cplusplus
	extern "C" {
#endif

tuple<VectorXf,double> lasso(MatrixXf, VectorXf, float, int, float, float, float, MPI_Comm);	

//float* lasso_admm (MatrixXd  A_in, int m, int n,  VectorXd b_in, float lambda, MPI_Comm comm);
	
#ifdef __cplusplus
 }
#endif

#endif
