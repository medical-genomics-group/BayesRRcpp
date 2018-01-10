#ifndef distributions_H
#define distributions_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>


Eigen::VectorXd dirichilet_rng(Eigen::VectorXd alpha);
double inv_gamma_rng(double shape,double scale);
double gamma_rng(double shape,double scale);
double inv_gamma_rate_rng(double shape,double rate);
double gamma_rate_rng(double shape,double rate);
double inv_scaled_chisq_rng(double dof,double scale);
double norm_rng(double mu, double sigma2);
double component_probs(double b,Eigen::VectorXd pi);
double categorical(Eigen::VectorXd probs);
double beta_rng(double a, double b);
double exp_rng(double a);
//double spike_slab_rng(double w,double lambda, double sigmaS);
#endif
