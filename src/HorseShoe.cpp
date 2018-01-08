#include <omp.h>
// [[Rcpp::plugins(openmp)]]
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
struct exponential_functor
{
  exponential_functor(){}

  const Scalar operator()(const Scalar& x) const{ return exp_rng(x); }

};


inline MatrixXd AtA(const MapMatd& A) {
  int n(A.cols());
  return MatrixXd(n,n).setZero().selfadjointView<Lower>()
                      .rankUpdate(A.adjoint());
}

template<typename Scalar>
struct inv_gamma_functor
{
  inv_gamma_functor(){}

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rng(1.0,x); }

};

//Functor to perform a elementwise draw from a conditional categorical distribution with vector of probabilities pi, coefficient beta and variance sigma
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
* Bayes R sampler Cholesky rank update
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
* B- integer lower than number of columns of X, number of blocks in which the effects beta will be conditiionally divided as p(beta_B|beta_\B,.)
*/
// [[Rcpp::export]]
void HorseshoeP(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,double sigma0, double v0E, double s02E, double v0G, double s02G, int B) {
  int flag;
  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
  flag=0;
  int N(Y.size());
  int M(X.cols());
  MatrixXd xtY(M,1);
  MatrixXd xtX(M,M);

  VectorXd ones(N);
  Map<MatrixXd> xM(X.data(),N,M);
  ////////////validate inputs
  if(B>M || B<=0) /////////we validate the number of blocks
  {
    std::cout<<"error: Number of blocks has to be a positive integer and smaller than the number of covariates in the model (columns of X) ";
    return;
  }
  if(max_iterations < burn_in || max_iterations<1 || burn_in<1) //validations related to mcmc burnin and iterations
  {
    std::cout<<"error: burn_in has to be a positive integer and smaller than the maximum number of iterations ";
    return;
  }
  if(sigma0 < 0 || v0E < 0 || s02E < 0 || v0G < 0||  s02G < 0 )//validations related to hyperparameters
  {
    std::cout<<"error: hyper parameters have to be positive";
    return;
  }
  /////end of declarations//////
  Eigen::initParallel();
  Eigen::setNbThreads(10);
  ones.setOnes();

  std::cout<<" computing XtX\n";
  std::chrono::high_resolution_clock::time_point startt= std::chrono::high_resolution_clock::now();
  xtX=AtA(xM);
  std::chrono::high_resolution_clock::time_point stopt= std::chrono::high_resolution_clock::now();
  auto durationt = std::chrono::duration_cast<std::chrono::seconds>( startt - stopt ).count();
  std::cout << "crossproduct was computed in: "<<durationt << "s\n";

#pragma omp parallel num_threads(2) shared(flag,q,M,N)
{
#pragma omp sections
{

  {

    scalar_normal_dist_op<double> randN;
    double mu;
    int b(M/B);
    MatrixXd mu_b(N,1);

    VectorXd mu_f(N);
    double sigmaE;
    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    VectorXd lambda(M);
    VectorXd v(M);
    double tau;
    VectorXd residues(M);
    MatrixXd beta(M,1);
    double eta;

    double sum_beta_sqr;

    VectorXd sample(2*M+5);

    int beginSegment;
    int endSegment;
    int blockNo;



    //beta.setRandom();
    beta=(xtX).colPivHouseholderQr().solve(X.transpose()*Y);
    mu=norm_rng(0,1);
    sigmaE=std::abs(norm_rng(0,1));
    eta=0.00001*std::abs(norm_rng(0,1));
    tau=std::abs(norm_rng(0,1));
    lambda=lambda.setRandom().cwiseAbs();
    v=v.setRandom().cwiseAbs();
    // residues=Y-mu*ones;


    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    residues= X*beta;

    for(int iteration=0; iteration < max_iterations; iteration++){
   //   std::cout <<beta.col(0) <<"\n";
      std::cout << "iteration: "<<iteration <<"\n";
   //   mu = norm_rng(1.0/(1.0/sigma0+(double)N/sigmaE)*(Y-residues).sum(), 1.0/(1.0/sigma0+(double)N/sigmaE));



      //lambda=(v.cwiseInverse()+((0.5/(tau*sigmaE))*beta.cwiseProduct(beta))).unaryExpr(inv_gamma_functor<double>());
      lambda=(v.cwiseInverse()+((0.5/(tau))*beta.cwiseProduct(beta))).unaryExpr(inv_gamma_functor<double>());



      tau= inv_gamma_rng(0.5*(M+1.0),1.0/eta+((0.5/sigmaE)*((lambda.cwiseInverse()).cwiseProduct(beta.cwiseProduct(beta))).sum()));
      tau=sigma0;

      blockNo=1;
      b=M/B;

      VectorXd temp(M);
      temp=(lambda*tau).cwiseInverse();

      for(int block=0; block < M;block+=b){

        beginSegment=block;
        endSegment=(block+b-1)>=(M-1)?(M-1):(block+b-1);
        //if uneven block splitting, we let the last block to have one more element
        if(M % B > 0 && (blockNo==B)){
          b=b+M%B;
        }
        blockNo+=1;

        if(beginSegment==0){
          mu_b=residues-((X.block(0,0,N,b)*beta.block(0,0,b,1)));

        }
        else{

          mu_b+=(-X.block(0,beginSegment,N,b)*beta.block(beginSegment,0,b,1));
        }


        beta.block(beginSegment,0,b,1)= mvnCoef_rng(1,
                   X.block(0,beginSegment,N,b).transpose()*(Y-mu_b-mu_f),
                   xtX.block(beginSegment,beginSegment,b,b),
                   temp.segment(beginSegment,b)*sigmaE,sigmaE); //check which one is the correct expression (substitute 1.0 with sigmaE)

        // beta.block(beginSegment,0,b,1) = (components.block(beginSegment,0,b,1).array() > 1e-10 ).select(beta.block(beginSegment,0,b,1), MatrixXd::Zero(b,1));
        mu_f+=X.block(0,beginSegment,N,b)*beta.block(beginSegment,0,b,1);
      }


      residues=mu_f;
      eta = inv_gamma_rng(1,1.0+1.0/tau);

      v=((lambda.cwiseInverse()).array()+1.0).unaryExpr(inv_gamma_functor<double>());
      //double temp=(Y.transpose()*( MatrixXd::Identity(N, N)-tau*X.transpose()*lambda.asDiagonal()*X)*Y);
      //sigmaE=inv_gamma_rng(N/2.0,temp/2.0);

      //std::cout <<beta.col(0) <<"\n";
      //sigmaE=inv_gamma_rng((N+M)*0.5,(Y-residues).squaredNorm()*0.5 + (beta.cwiseProduct(beta).cwiseProduct(temp)).sum()* 0.5);
      sigmaE=inv_scaled_chisq_rng(v0E+N,((Y-residues).squaredNorm()+v0E*s02E)/(v0E+N));

      double term;
      term= Y.transpose()*X*temp.asDiagonal()*X.transpose()*Y;
      //sigmaE=inv_gamma_rng(N*0.5, 0.5*Y.squaredNorm() - 0.5*term);
      //std::cout << 0.5*(((Y.transpose()*X).cwiseProduct(Y.transpose()*X)).cwiseProduct(temp).sum())<<"\n";
      std::cout << sigmaE<<"\n";
      sum_beta_sqr= (1.0/N)*residues.squaredNorm() - pow(residues.mean(),2);
      if(iteration >= burn_in)
      {
        sample<< iteration,mu,beta,sigmaE,tau,lambda, sum_beta_sqr;
        q.enqueue(sample);
      }

    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
    std::cout << "duration: "<<duration << "s\n";
    flag=1;
  }
#pragma omp section
{
  bool queueFull;
  queueFull=0;
  std::ofstream outFile;
  outFile.open(outputFile);
  VectorXd sampleq(2*M+5);
  IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
  outFile<< "iteration,"<<"mu,";
  for(unsigned int i = 0; i < M; ++i){
    outFile << "beta[" << (i+1) << "],";

  }
  outFile<<"sigmaE,"<<"tau,";
  for(unsigned int i = 0; i < M; ++i){
    outFile << "lambda[" << (i+1) << "],";
  }
  outFile<<"EV\n";
  while(!flag ){
    if(q.try_dequeue(sampleq))
      outFile<< sampleq.transpose().format(CommaInitFmt) << "\n";
  }
  //
}

}
}

}


/*** R
M=2000
N=2000
B=matrix(rnorm(M,sd=sqrt(0.5/M)),ncol=1)
  X <- matrix(rnorm(M*N), N, M); var(X[,1])
    G <- X%*%B; var(G)
      Y=X%*%B+rnorm(N,sd=0.001); var(Y)
        Y=scale(Y)
        X=scale(X)
       HorseshoeP("./test2.csv",1, 000,500 ,1000,X, Y,0.0001,0.01,0.01,0.01,0.01,100)
        library(readr)
        tmp <- read_csv("./test2.csv")
#names(tmp)
#plot(tmp$sigmaG); mean(tmp$sigmaG)
        plot(B,colMeans(tmp[,grep("beta",names(tmp))]))
        lines(B,B)
        abline(h=0)
        G <- X%*%B;
      var(G)
        mean(tmp$EV)
        1-var(G)
        mean(tmp$sigmaE)
        plot(tmp$sigmaE)
        plot(tmp$mu)
        plot(tmp$tau)
        hist(as.matrix(tmp[,grep("beta",names(tmp))]))
        colMeans(tmp[,grep("lambda",names(tmp))])

        plot(B,colMeans(tmp[,grep("beta",names(tmp))]))
        lines(B,B)

        */

