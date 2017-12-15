/**
 *Code modified by Daniel Trejo Baños for allowing conditional draws from a multivariate normal (completing the square). Below the original comments and license
 *
 *
 * Multivariate Normal distribution sampling using C++11 and Eigen matrices.
 *
 * This is an extension of the multivariate normal simulation used in https://github.com/beniz/eigenmvn
 *
 * I included functionality necessary to sample from conditional gaussian random fields. Daniel Trejo Banos
 *
 * here is the original licence and message :
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

//TODO clean and test
//verify the correct way to sample from conditional coef
#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>
#include "MultVar.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace RcppEigen;
using Eigen::Matrix;
using Eigen::MatrixXd;                  // variable size matrix, double precision

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//
// [[Rcpp::export]]
Eigen::MatrixXd sumDiagonal(Eigen::MatrixXd Ma,Eigen::VectorXd va)
{
   Ma.diagonal()+=va;
  return Ma;
}
Eigen::MatrixXd sumIdentity(Eigen::MatrixXd Ma)
{
 for(int i=0;i<Ma.diagonalSize();i++)
  Ma(i,i)+=1;
  return Ma;
}

namespace  Eigen{
namespace internal {
template<typename Scalar>
struct scalar_normal_dist_op
{
  static std::mt19937 rng;                        // The uniform pseudo-random algorithm
  mutable std::normal_distribution<Scalar> norm; // gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

    template<typename Index>
    inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
    inline void seed(const uint64_t &s) { rng.seed(s); }
};

template<typename Scalar>
std::mt19937 scalar_normal_dist_op<Scalar>::rng;

template<typename Scalar>
struct functor_traits<scalar_normal_dist_op<Scalar> >
{ enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };

} // end namespace internal


/**
Find the eigen-decomposition of the covariance matrix
and then store it for sampling from a multi-variate normal
*/
template<typename Scalar>
class EigenMultivariateNormal
{
  Matrix<Scalar,Dynamic,Dynamic> _covar;
  Matrix<Scalar,Dynamic,Dynamic> _transform;
  Matrix< Scalar, Dynamic, 1> _mean;
  internal::scalar_normal_dist_op<Scalar> randN; // Gaussian functor
  bool _use_cholesky;
  SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> > _eigenSolver; // drawback: this creates a useless eigenSolver when using Cholesky decomposition, but it yields access to eigenvalues and vectors

public:

  EigenMultivariateNormal(const Matrix<Scalar,Dynamic,1>& mean,const Matrix<Scalar,Dynamic,Dynamic>& covar,
                          const bool use_cholesky=false,const uint64_t &seed=std::mt19937::default_seed)
    :_use_cholesky(use_cholesky)
  {
    randN.seed(seed);
    setMean(mean);
    setCovar(covar);
  }

  void setMean(const Matrix<Scalar,Dynamic,1>& mean) { _mean = mean; }
  void setCovar(const Matrix<Scalar,Dynamic,Dynamic>& covar)
  {
    _covar = covar;

    // Assuming that we'll be using this repeatedly,
    // compute the transformation matrix that will
    // be applied to unit-variance independent normals

    if (_use_cholesky)
    {
      Eigen::LLT<Eigen::Matrix<Scalar,Dynamic,Dynamic> > cholSolver(_covar);
      // We can only use the cholesky decomposition if
      // the covariance matrix is symmetric, pos-definite.
      // But a covariance matrix might be pos-semi-definite.
      // In that case, we'll go to an EigenSolver
      if (cholSolver.info()==Eigen::Success)
      {
        // Use cholesky solver
        _transform = cholSolver.matrixL();
      }
      else
      {
        throw std::runtime_error("Failed computing the Cholesky decomposition. Use solver instead");
      }
    }
    else
    {
      _eigenSolver = SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> >(_covar);
      _transform = _eigenSolver.eigenvectors()*_eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    }
  }

  /// Draw nn samples from the gaussian and return them
  /// as columns in a Dynamic by nn matrix
   Matrix<Scalar,Dynamic,-1> samples(int nn)
  {
    return (_transform * Matrix<Scalar,Dynamic,-1>::NullaryExpr(_covar.rows(),nn,randN)).colwise() + _mean;
  }
};
// end class EigenMultivariateNormal
//conditional multivariate normal
template<typename Scalar>
class EigenMultivariateNormalCoef
{
  Eigen::Matrix<Scalar,Dynamic,Dynamic> _covar;
  Eigen::LLT<Eigen::Matrix<Scalar,Dynamic,Dynamic> >_transform;
  Matrix< Scalar, Dynamic, 1> _mean;
  internal::scalar_normal_dist_op<Scalar> randN; // Gaussian functor
  bool _use_cholesky;
  SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> > _eigenSolver; // drawback: this creates a useless eigenSolver when using Cholesky decomposition, but it yields access to eigenvalues and vectors

public:

  EigenMultivariateNormalCoef(const Matrix<Scalar,Dynamic,1>& xty,const Matrix<Scalar,Dynamic,Dynamic>& xtx,const VectorXd d,
                          const bool use_cholesky=false,const uint64_t &seed=std::mt19937::default_seed)
    :_use_cholesky(use_cholesky)
  {
    randN.seed(seed);
    setCovar(xtx,d);
    setMean(xty);
  }

  void setMean(const Matrix<Scalar,Dynamic,1>& xty) { _mean = xty; }
  void setCovar(const Matrix<Scalar,Dynamic,Dynamic>& xtx,const VectorXd d)
  {
    _covar =sumDiagonal(xtx,d);

    // Assuming that we'll be using this repeatedly,
    // compute the transformation matrix that will
    // be applied to unit-variance independent normals

    if (_use_cholesky)
    {
      Eigen::LLT<Eigen::Matrix<Scalar,Dynamic,Dynamic> > cholSolver(_covar);
      // We can only use the cholesky decomposition if
      // the covariance matrix is symmetric, pos-definite.
      // But a covariance matrix might be pos-semi-definite.
      // In that case, we'll go to an EigenSolver
      if (cholSolver.info()==Eigen::Success)
      {
        // Use cholesky solver
        _transform = cholSolver;
      }
      else
      {
        throw std::runtime_error("Failed computing the Cholesky decomposition. Use solver instead");
      }
    }
    else
    {
     // _eigenSolver = SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> >(_covar);
      //_transform = _eigenSolver.eigenvectors()*_eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    }
  }

  /// Draw nn samples from the gaussian and return them
  /// as columns in a Dynamic by nn matrix
  Matrix<Scalar,Dynamic,Dynamic> samples(int nn)
  {
    //TODO verify if I have to solve the linear system instead
    return  _transform.solve( Matrix<Scalar,Dynamic,-1>::NullaryExpr(_covar.rows(),1,randN))+_transform.solve(_mean);
  }
};
// end class EigenMultivariateNormalCoef
template<typename Scalar>
class EigenMultivariateNormalCoefAug
{
  Eigen::Matrix<Scalar,Dynamic,Dynamic> _covar;
  Eigen::Matrix< Scalar, Dynamic, 1> _mean;
  internal::scalar_normal_dist_op<Scalar> randN; // Gaussian functor
  bool _use_cholesky;
  SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> > _eigenSolver; // drawback: this creates a useless eigenSolver when using Cholesky decomposition, but it yields access to eigenvalues and vectors

public:

  EigenMultivariateNormalCoefAug(const Matrix<Scalar,Dynamic,1>& y,const Matrix<Scalar,Dynamic,Dynamic>& x,const VectorXd d,
                              const bool use_cholesky=false,const uint64_t &seed=std::mt19937::default_seed)
    :_use_cholesky(use_cholesky)
  {
    randN.seed(seed);
    setCovar(y,x,d);
    setMean(y);
  }

  void setMean(const Matrix<Scalar,Dynamic,1>& y) { _mean = y; }
  void setCovar(const Matrix<Scalar,Dynamic,1>& y,const Matrix<Scalar,Dynamic,Dynamic>& x,const VectorXd d)
  {
     Matrix<Scalar,Dynamic,Dynamic> u;
     u=(d.array()*Matrix<Scalar,Dynamic,-1>::NullaryExpr(d.rows(),1,randN).array()).matrix();
    _covar =u + d.asDiagonal()*x.transpose()*(sumIdentity(x*d.asDiagonal()*x.transpose())).lu().solve( y-x*u-(Matrix<Scalar,Dynamic,-1>::NullaryExpr(x.rows(),1,randN)).matrix());
  }

  /// Draw nn samples from the gaussian and return them
  /// as columns in a Dynamic by nn matrix
  Matrix<Scalar,Dynamic,Dynamic> samples(int nn)
  {
    //TODO verify if I have to solve the linear system instead
    return  _covar;
  }
};
// end class EigenMultivariateNormalAug


} // end namespace Eigen
// [[Rcpp::export]]
Eigen::MatrixXd mvn_rng(int nn,const Eigen::MatrixXd mean,const Eigen::MatrixXd covar)
{
  Eigen::EigenMultivariateNormal<double> normX_cholesk(mean,covar,true);
  return normX_cholesk.samples(nn);
}
// [[Rcpp::export]]
Eigen::MatrixXd mvnCoef_rng(int nn,const Eigen::MatrixXd xty,const Eigen::MatrixXd xtx,const  Eigen::VectorXd d)
{
  Eigen::EigenMultivariateNormalCoef<double> normX_cholesk(xty,xtx,d, true);
  return normX_cholesk.samples(nn);
}
// [[Rcpp::export]]
Eigen::MatrixXd mvnCoef_rngAug(int nn,const Eigen::MatrixXd y,const Eigen::MatrixXd x,const  Eigen::VectorXd d)
{
  Eigen::EigenMultivariateNormalCoefAug<double> normX_cholesk(y,x,d, true);
  return normX_cholesk.samples(nn);
}
//sum diagonal matrix to a matrix

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//


/*** R
M=1000
N=100
B=matrix(rnorm(M),1)
X <- matrix(rnorm(M*N), N, M)
Y=X%*%t(B)
Y=scale(Y)
X=scale(X)

xtx=t(X)%*%X
xty=t(X)%*%Y
d= rep(1,M)
sumDiagonal(xtx,d)

#mvn_rng(100,Y,xtx)

#mvnCoef_rng(1,xty,xtx,d)
#mvnCoef_rngAug(1,Y,X,d)
*/


