// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Core>
#include <random>
#include "distributions.h"
#include "MultVar.h"
#include "concurrentqueue.h"

// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace RcppEigen;
using namespace Eigen;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseVector;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::Map;
using Eigen::Upper;
typedef Map<MatrixXd> MapMatd;


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
struct categorical_functor
{
  categorical_functor(const Eigen::VectorXd& pi,const Scalar& sigmaG) : m_b(sigmaG),m_c(pi){}

  const Scalar operator()(const Scalar& x) const{ return component_probs(x,m_c,m_b); }
  Scalar  m_b;
  Eigen::VectorXd m_c;
};
//Functor perform a componentwise draw from a categorical distribution with parameters pi
template<typename Scalar>
struct categorical_init
{
  categorical_init(const Eigen::VectorXd& pi) : m_c(pi){}

  const Scalar operator()(const Scalar& x) const{ return categorical(m_c); }
  Eigen::VectorXd m_c;
};

/*
* Bayes R sampler
* outputFile- The file in which the samples aftare burnin will be stored
* seed- random seed
* max_iterations- total of number of samples taken.
* burn_in - integer leq than max_iterations, number of samples used for burn in, after which, al samples will  be stored in the outputFile
* thinning- thinning regime, not implemented
* X- matrix of snp markers, or covariates of interest
* Y- vector of response variates, must have the same number of rows as X
* sigma0- variance of the zero-centered normal prior over the intercept
* v0E- degrees of  freedom of the prior inverse scaled chi-squared distribution over residues variance
* s02E - scale parameter of the prior inverse scaled chi-squared distribution over residues variance
* v0G- degrees of freedom of the prior inverse scaled chi-squared distribution over genetic effects variance
* s02G- scale parameter of the prior inverse scaled chi-squared distribution over genetic effects variance
*/
// [[Rcpp::export]]
void BRV2PPGroups(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning,Eigen::VectorXd mu, Eigen::MatrixXd beta, Eigen::VectorXd sigmaE, Eigen::MatrixXd X,Eigen::VectorXd y) {
  int flag;
  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
  flag=0;
  int N(X.rows());
  int M(X.cols());
  int S(beta.rows());
  Eigen::VectorXd sample(3*N);
  Eigen::initParallel();
  Eigen::setNbThreads(10);


#pragma omp parallel num_threads(2) shared(flag,M,N)
{
#pragma omp sections
{

  {
    VectorXd y_lin(N);
    VectorXd y_pred(N);
    VectorXd epsilon(N);
    scalar_normal_dist_op<double> randN;

    for(int iteration=0; iteration < S; iteration++){

             y_lin = mu(iteration)+(X*beta.row(iteration)).array();
             y_pred = y_lin + sqrt(sigmaE(iteration))*(Eigen::Matrix<double,Dynamic,-1>::NullaryExpr(N,1,randN));
             epsilon= y-y_lin;
            sample<< y_lin,y_pred,epsilon;
            q.enqueue(sample);
    }
    flag=1;
  }
#pragma omp section
{
  bool queueFull;
  queueFull=0;
  std::ofstream outFile;
  outFile.open(outputFile);
  VectorXd sampleq(3*N);
  IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");

  for(unsigned int i = 0; i < N; ++i){
    outFile << "y_lin[" << (i+1) << "],";
  }
  for(unsigned int i = 0; i < (N); ++i){
    outFile << "y_pred[" << (i+1) << "],";
  }
  for(unsigned int i = 0; i < (N-1); ++i){
    outFile << "epsilon[" << (i+1) << "],";
  }
  outFile << "epsilon[" << (N-1) << "]";
  outFile<<"\n";

  while(!flag ){
    if(q.try_dequeue(sampleq))
      outFile<< sampleq.transpose().format(CommaInitFmt) << "\n";
  }
}

}
}

}

