#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>
#include <math.h>
#include "distributions.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace RcppEigen;
using Eigen::VectorXd;

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
double inv_gamma_rng(double shape,double scale){
  return (1.0 / R::rgamma(shape, 1.0/scale));
}
double gamma_rng(double shape,double scale){
  return R::rgamma(shape, scale);
}
double inv_gamma_rate_rng(double shape,double rate){
  return 1.0 / gamma_rate_rng(shape, rate);
}
double gamma_rate_rng(double shape,double rate){
  return R::rgamma(shape,1.0/rate);
}
// [[Rcpp::export]]
double inv_scaled_chisq_rng(double dof,double scale){
  return inv_gamma_rng(0.5*dof, 0.5*dof*scale);
}
double norm_rng(double mean,double sigma2){
  return R::rnorm(mean,std::sqrt((double)sigma2));
}
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

inline double bernoulli_rng(double probs0,double probs1,double cats0,double cats1){
  double p;
  p=R::runif(0,1);
  if(p<= probs0/(probs0+probs1))
    return cats0;
  else
    return cats1;
}

double beta_rng(double a,double b){
  return R::rbeta(a,b);
}
double exp_rng(double a){
  return R::rexp(a);
}
