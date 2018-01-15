#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>
#include <math.h>
#include "distributions.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace RcppEigen;
using Eigen::VectorXd;
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
Eigen::VectorXd dirichilet_rng(Eigen::VectorXd alpha) {
  int len;
  len=alpha.size();
  VectorXd result(len);
  for(int i=0;i<len;i++)
    result[i]=R::rgamma(alpha[i],1);
  result/=result.sum();
  return result;
}
// [[Rcpp::export]]
double inv_gamma_rng(double shape,double scale){
  return (1.0 / R::rgamma(shape, 1.0/scale));
}
// [[Rcpp::export]]
double gamma_rng(double shape,double scale){
  return R::rgamma(shape, scale);
}
// [[Rcpp::export]]
double inv_gamma_rate_rng(double shape,double rate){
  return 1.0 / R::rgamma(shape, rate);
}
// [[Rcpp::export]]
double gamma_rate_rng(double shape,double rate){
  return R::rgamma(shape,1.0/rate);
}

// [[Rcpp::export]]
double inv_scaled_chisq_rng(double dof,double scale){
  return inv_gamma_rng(0.5*dof, 0.5*dof*scale);
}
// [[Rcpp::export]]
double norm_rng(double mean,double sigma2){
  return R::rnorm(mean,std::sqrt((double)sigma2));
}
// [[Rcpp::export]]
double component_probs(double b2,Eigen::VectorXd pi){
  double sum;
  double p;
  p=R::runif(0,1);
  sum= pi[0]*exp((-0.5*b2)/(5e-2))/sqrt(5e-2)+pi[1]*exp((-0.5*b2));
  if(p<=(pi[0]*exp((-0.5*b2)/(5e-2))/sqrt(5e-2))/sum)
    return 5e-2;
  else
    return 1;
}

double categorical(Eigen::VectorXd probs){
  double p;

  p=R::runif(0,1);
  if(p<= probs[0]/(probs[0]+probs[1]))
    return 5e-2;
  else
    return 1;



}

double beta_rng(double a,double b){
  return R::rbeta(a,b);
}
double exp_rng(double a){
  return R::rexp(a);
}
// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
#tmp<-dirichilet_rng(c(1.0,2.0,3.0,4.0))
#tmp
#inv_gamma_rng(0.05,0.05)
#inv_scaled_chisq_rng(1,0.1)
#norm_rng(0,1)
#component_probs(0.01,c(0.25,0.25,0.25,0.25),1)
spike_slab_rng(0.5,0,0.02)
*/
