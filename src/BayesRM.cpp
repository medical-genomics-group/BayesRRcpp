#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Core>
#include <random>
#include "distributions.h"
#include "MultVar.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace RcppEigen;
using namespace Eigen;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseVector;
// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//
template<typename Scalar>
struct categorical_functor
{
  categorical_functor(const Eigen::VectorXd& pi,const Scalar& sigmaG) : m_b(sigmaG),m_c(pi){}

  const Scalar operator()(const Scalar& x) const{ return component_probs(x,m_c,m_b); }
  Scalar  m_b;
  Eigen::VectorXd m_c;
};
template<typename Scalar>
struct categorical_init
{
  categorical_init(const Eigen::VectorXd& pi) : m_c(pi){}

  const Scalar operator()(const Scalar& x) const{ return categorical(m_c); }
  Eigen::VectorXd m_c;
};


// [[Rcpp::export]]
Rcpp::List BayesRSamplerM(int seed, int max_iterations, int burn_in,int thinning,Eigen::MatrixXd X, Eigen::MatrixXd Y,double v0,double s02) {
  int N;
  int M;
  int Q;
  VectorXd mu(Y.cols());
  VectorXd sigmaG(Y.cols());
  VectorXd sigmaE(Y.cols());
  int m0;
  std::random_device rd;
  std::mt19937 gen(rd());

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::normal_distribution<> norm(0,1);
  VectorXd priorPi(4,Y.cols());
  MatrixXd pi(4,Y.cols());
  VectorXd cVa(4);
  VectorXd invsqrtcVa(4);
  MatrixXd residues(Y.rows(),Y.cols());
  VectorXd components(X.cols(),Y.cols());
  MatrixXd beta(X.cols(),Y.cols());
  MatrixXd betaL(beta.rows()*beta.cols(),max_iterations);
  MatrixXd componentsL(beta.rows()*beta.cols(),max_iterations);
  MatrixXd sigmaGL(Y.rows(),max_iterations);
  MatrixXd sigmaEL(Y.rows(),max_iterations);
  MatrixXd muL(Y.rows(),max_iterations);
  MatrixXd piL(4,max_iterations);
  MatrixXd xtX(X.cols(),X.cols());
  MatrixXd xtY(X.cols(),Q);
  MatrixXd v(4,Y.cols());



  M=X.cols();
  N=Y.rows();
  Q=Y.cols();
  xtX=MatrixXd(M,M).setZero().selfadjointView<Lower>().rankUpdate(X.adjoint());
  xtY=X.transpose()*Y;
  priorPi.setOnes();
  priorPi*=0.25;
  cVa[0] = 0;
  cVa[1] = 0.0001;
  cVa[2] = 0.001;
  cVa[3] = 0.01;
  invsqrtcVa[0]=0;
  invsqrtcVa.segment(1,3)=-1*(cVa.segment(1,3)).array().square().cwiseInverse();

  beta.setRandom();
  mu.setRandom();
  sigmaE=sigmaE.setRandom().cwiseAbs();
  sigmaG=sigmaG.setRandom().cwiseAbs();
  for(int i=0;i<Q;i++)
    pi.col(i)=dirichilet_rng(priorPi);
  components.unaryExpr(categorical_init<double>(priorPi));
  residues=Y-X*beta;

  for(int iteration=0; iteration < max_iterations; iteration++){
    std::cout << "iteration: "<<iteration <<"\n";

    for(int i=0;i<Q;i++){
      mu(i) = norm_rng((1/(double)N)*residues.col(i).sum(), sigmaE(i)/(double)N);
      components.col(i)= beta.col(i).unaryExpr(categorical_functor<double>(pi.col(i),sigmaG(i)));
      beta.col(i)= mvnCoef_rng(1,xtY.col(i),xtX,sigmaG(i)*components.col(i));
      residues.col(i)=Y.col(i)-X*beta.col(i);
      m0=(components.col(i).array()>1e-10).count();
      sigmaG(i)=inv_scaled_chisq_rng(v0+m0,(beta.col(i).array().pow(2).sum()+v0*s02)/(v0+m0));
      sigmaE(i)=inv_scaled_chisq_rng(v0+N,(((residues.col(i).array()-mu(i)).array().pow(2)).sum()+v0*s02)/(v0+N));
      v(0,i)=priorPi[0]+(components.col(i).array()==cVa[0]).count();
      v(1,i)=priorPi[1]+(components.col(i).array()==cVa[1]).count();
      v(2,i)=priorPi[2]+(components.col(i).array()==cVa[2]).count();
      v(3,i)=priorPi[3]+(components.col(i).array()==cVa[3]).count();
      pi.col(i)=dirichilet_rng(v.col(i));
    }

    Map<MatrixXd> betaR(beta.data(),beta.rows()*beta.cols(),1);///
    betaL.col(iteration)=betaR;
    sigmaGL.col(iteration)=sigmaG;
    sigmaEL.col(iteration)=sigmaE;
    muL.col(iteration)=mu;
    Map<MatrixXd> piR(pi.data(),pi.rows()*pi.cols(),1);
    piL.col(iteration)=piR;//
    Map<MatrixXd> componentsR(components.data(),components.rows()*components.cols(),1);
    componentsL.col(iteration)=componentsR;//
  }
  return Rcpp::List::create(Rcpp::Named("beta")=betaL.transpose(),
                            Rcpp::Named("sigmaG")=sigmaGL.transpose(),
                            Rcpp::Named("sigmaE")=sigmaEL.transpose(),
                            Rcpp::Named("pi")=piL.transpose(),
                            Rcpp::Named("componets")=componentsL.transpose(),
                            Rcpp::Named("mu")=muL.transpose());
}


/*** R
M=100
N=10000
Q=2
B=matrix(rnorm(M*Q,sd=0.1),ncol=Q)
B[sample(1:M,30),]=0
X <- matrix(rnorm(M*N), N, M)
Y=X%*%B+matrix(rnorm(N*Q,sd=0.01),ncol=Q)
Y=scale(Y)
X=scale(X)

tmp<-BayesRSamplerM(3000, 1000, 1,1,X, Y,1,1)
plot(c(B[,1],B[,2]),colMeans(tmp$beta))
*/
