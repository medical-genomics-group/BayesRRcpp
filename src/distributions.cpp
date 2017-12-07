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
  return (1.0 / R::rgamma(shape, 1.0 / scale));
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
double component_probs(double b,Eigen::VectorXd pi,double sigmaG){
  double sum;
  double p;
  double b2;
  p=R::runif(0,1);
  b2=b*b;
  sum= ((abs(b)==0)?1:0)*pi[0]+pi[1]*(1.0/sqrt(0.0001))*exp((-0.5*b2)/(0.0001*sigmaG)) +pi[2]*(1.0/sqrt(0.001))*exp((-0.5*b2)/(0.001*sigmaG))+pi[3]*(1.0/sqrt(0.01))*exp((-0.5*b2)/(0.01*sigmaG));
  //not pretty but will save space if done concurrently, binary search
  if(p<=((abs(b)==0)?1:0)*pi[0]/sum+pi[1]*(1.0/sqrt(0.0001))*exp((-0.5*b2)/(0.0001*sigmaG))/sum){
    if(p<=((abs(b)==0)?1:0)*pi[0]/sum)
      return 0;
    else
      return 0.0001;
  }
  else
  {
    if(p<=((abs(b)==0)?1:0)*pi[0]/sum+pi[1]*(1.0/sqrt(0.0001))*exp((-0.5*b2)/(0.0001*sigmaG))/sum+pi[2]*(1.0/sqrt(0.001))*exp((-0.5*b2)/(0.001*sigmaG))/sum)
      return 0.001;
    else
      return  0.01;
  }
}

double categorical(Eigen::VectorXd probs){
  double p;

  p=R::runif(0,1);
  if(p<= probs[0]+probs[1]){
    if(p<=probs[0])
      return 0;
    else
      return 0.0001;

  }
  else{
    if(p<=probs[2])
      return 0.001;
    else
      return 0.01;
  }

}

double beta_rng(double a,double b){
  return R::rbeta(a,b);
}
// [[Rcpp::export]]
double spike_slab_rng(double w,double lambda,double sigmaS){
   double p;
   double sum;
   double lk1;
   double lk0;
   double prob;
   p=R::runif(0,1);
   lk1= log(w)-0.5*lambda*lambda/sigmaS;
   lk0= (lambda<=1e-10?1:0)*log(1-w);
   prob=1/(exp(lk1-lk0));
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
