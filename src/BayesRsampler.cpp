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
Rcpp::List BayesRSampler(int seed, int max_iterations, int burn_in,int thinning,Eigen::MatrixXd X, Eigen::VectorXd Y,double v0,double s02) {
  int N;
  int M;
  double mu;
  double sigmaG;
  double sigmaE;
  int m0;
  std::random_device rd;
  std::mt19937 gen(rd());

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::normal_distribution<> norm(0,1);
  VectorXd priorPi(4);
  VectorXd pi(4);
  VectorXd cVa(4);
  VectorXd invsqrtcVa(4);
  VectorXd residues;
  VectorXd components(X.cols());
  MatrixXd beta(X.cols(),1);
  MatrixXd betaL(X.cols(),max_iterations);
  MatrixXd componentsL(X.cols(),max_iterations);
  MatrixXd sigmaGL(1,max_iterations);
  MatrixXd sigmaEL(1,max_iterations);
  MatrixXd muL(1,max_iterations);
  MatrixXd piL(4,max_iterations);
  MatrixXd xtX(X.cols(),X.cols());
  MatrixXd xtY(X.cols(),1);



  VectorXd v(4);
  M=X.cols();
  N=Y.rows();
  xtX=X.transpose()*X;
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
  mu=norm(gen);
  sigmaE=abs(norm_rng(0,1));
  sigmaG=abs(norm_rng(0,1));
  pi=dirichilet_rng(priorPi);
  components.unaryExpr(categorical_init<double>(priorPi));
  residues=Y-X*beta;

  for(int iteration=0; iteration < max_iterations; iteration++){
    std::cout << "iteration: "<<iteration <<"\n";
    mu = norm_rng((1/(double)N)*residues.sum(), sigmaE/(double)N);
    components= beta.unaryExpr(categorical_functor<double>(pi,sigmaG));
    beta= mvnCoef_rng(1,xtY,xtX,sigmaG*components);
    residues=Y-X*beta;
    m0=(components.array()>1e-10).count();
    sigmaG=inv_scaled_chisq_rng(v0+m0,(beta.array().pow(2).sum()+v0*s02)/(v0+m0));
    sigmaE=inv_scaled_chisq_rng(v0+N,(((residues.array()-mu).array().pow(2)).sum()+v0*s02)/(v0+N));
    v(0)=priorPi[0]+(components.array()==cVa[0]).count();
    v(1)=priorPi[1]+(components.array()==cVa[1]).count();
    v(2)=priorPi[2]+(components.array()==cVa[2]).count();
    v(3)=priorPi[3]+(components.array()==cVa[3]).count();
    pi=dirichilet_rng(v);
    betaL.col(iteration)=beta;
    sigmaGL(1,iteration)=sigmaG;
    sigmaEL(1,iteration)=sigmaE;
    muL(1,iteration)=mu;
    piL.col(iteration)=pi;
    componentsL.col(iteration)=components;
  }
    return Rcpp::List::create(Rcpp::Named("beta")=betaL.transpose(),
                              Rcpp::Named("sigmaG")=sigmaGL.transpose(),
                              Rcpp::Named("sigmaE")=sigmaEL.transpose(),
                              Rcpp::Named("pi")=piL.transpose(),
                              Rcpp::Named("componets")=componentsL.transpose(),
                              Rcpp::Named("mu")=muL.transpose());
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
M=100
N=1000
B=matrix(rnorm(M,sd=0.1),ncol=1)
B[sample(1:100,30),1]=0
  X <- matrix(rnorm(M*N), N, M)
  Y=X%*%B+rnorm(N,sd=0.1)
  Y=scale(Y)
  X=scale(X)

tmp<-BayesRSampler(3000, 11000, 1,1,X, Y,0.01,0.01)
plot(B,rowMeans(tmp$beta))

  */
