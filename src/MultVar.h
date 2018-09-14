/**
 * Code modified by Daniel Trejo Baños for allowing conditional draws from a multivariate normal (completing the square). Below the original comments and license
 *
 *
 * Multivariate Normal distribution sampling using C++11 and Eigen matrices.
 *
 * This is taken from http://stackoverflow.com/questions/16361226/error-while-creating-object-from-templated-class
 * (also see http://lost-found-wandering.blogspot.fr/2011/05/sampling-from-multivariate-normal-in-c.html)
 *
 * I have been unable to contact the original author, and I've performed
 * the following modifications to the original code:
 * - removal of the dependency to Boost, in favor of straight C++11;
 * - ability to choose from Solver or Cholesky decomposition (supposedly faster);
 * - fixed Cholesky by using LLT decomposition instead of LDLT that was not yielding
 *   a correctly rotated variance
 *   (see this http://stats.stackexchange.com/questions/48749/how-to-sample-from-a-multivariate-normal-given-the-pt-ldlt-p-decomposition-o )
 */

/**
* Copyright (c) 2014 by Emmanuel Benazera beniz@droidnik.fr, All rights reserved.
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3.0 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library.
*/
#ifndef MultVar_H
#define MultVar_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>

Eigen::MatrixXd mvnCoef_rng(int nn,const Eigen::MatrixXd xty,const Eigen::MatrixXd xtx,const  Eigen::VectorXd d,double sigma);
#endif
