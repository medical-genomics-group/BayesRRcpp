#ifndef MultVar_H
#define MultVar_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>

class EigenMultivariateNormalCoefAug;
Eigen::MatrixXd mvnCoef_rngAug(int nn,const Eigen::MatrixXd y,const Eigen::MatrixXd x,const  Eigen::VectorXd d);
#endif
