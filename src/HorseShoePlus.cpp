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
  inv_gamma_functor(const Scalar& vd):m_a(vd){}

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rate_rng(0.5 + 0.5*m_a,x); }
  Scalar  m_a;
};

template<typename Scalar>
struct gamma_functor
{
  gamma_functor(const Scalar& vd):m_a(vd){}

  const Scalar operator()(const Scalar& x) const{ return gamma_rng(0.5 + 0.5*m_a,x); }
  Scalar  m_a;
};

template<typename Scalar>
struct inv_gamma_functor_init
{
  inv_gamma_functor_init(){}

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rate_rng(0.5,x); }

};

template<typename Scalar>
struct inv_gamma_functor_init_v
{
  inv_gamma_functor_init_v(const Scalar& vd):m_a(vd){}

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rate_rng(0.5*m_a,m_a*x); }
  Scalar  m_a;
};

/*
* Horseshoe sampler
* outputFile- The file in which the samples aftare burnin will be stored
* seed- random seed
* max_iterations- total of number of samples taken.
* burn_in - integer leq than max_iterations, number of samples used for burn in, after which, al samples will  be stored in the outputFile
* thinning- thinning regime, not implemented
* X- matrix of snp markers, or covariates of interest
* Y- vector of response variates, must have the same number of rows as X
* A- variance of the student-t prior over tau
* v0E- degrees of  freedom of the prior inverse scaled chi-squared distribution over residues variance
* s02E - scale parameter of the prior inverse scaled chi-squared distribution over residues variance
* vL- degrees of freedom of the student t prior over local parameters lambda, vL=1 gives a cauchy prior
* vT- degrees of freedom of the student t prior over global parameter tau, vG=1 gives a cauchy prior
* B- integer lower than number of columns of X, number of blocks in which the effects beta will be conditiionally divided as p(beta_B|beta_\B,.)
*/
// [[Rcpp::export]]
void HorseshoePlus(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,double A, double v0E, double s02E, double vL, double vT, int B) {
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
  if(A < 0 || v0E < 0 || s02E < 0 || vL < 0||  vT < 0 )//validations related to hyperparameters
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
    VectorXd phi(M);
    VectorXd chi(M);
    double tau;
    VectorXd residues(M);
    MatrixXd beta(M,1);
    double eta;

    double sum_beta_sqr;

    VectorXd sample(2*M+5);

    int beginSegment;
    int endSegment;
    int blockNo;
    double c2=0.1;


    //beta=(xtX).colPivHouseholderQr().solve(X.transpose()*Y);
    beta.setRandom();
    mu=norm_rng(0,1);
    sigmaE=0.5;
    std::cout<< "initial SigmaE " << sigmaE<<"\n";
    eta=inv_gamma_rate_rng(0.5,1/pow(A,2));
    std::cout<< "initial eta " << eta<<"\n";
    tau=inv_gamma_rate_rng(0.5*vT,vT/eta);

   // tau=1/A;
    std::cout<< "initial tau " << tau<<"\n";
    chi=(chi.setOnes().array()).unaryExpr(inv_gamma_functor<double>(0));
    phi=(phi.setOnes().array()/chi.array()).unaryExpr(inv_gamma_functor<double>(0));
    v=(v.setOnes().array()/phi.array()).unaryExpr(inv_gamma_functor<double>(0));
    //std::cout<< "initial v" << eta;
    lambda=v.unaryExpr(inv_gamma_functor_init_v<double>(vL));
    //std::cout<< "initial lambda" << lambda;



    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    residues= X*beta;

    for(int iteration=0; iteration < max_iterations; iteration++){
   //   std::cout <<beta.col(0) <<"\n";
      std::cout << "iteration: "<<iteration <<"\n";
      lambda=(vL*v.cwiseInverse()+(0.5*beta.cwiseProduct(beta)*(1.0/(tau*sigmaE)))).unaryExpr(inv_gamma_functor<double>(vL));

      tau= inv_gamma_rate_rng(0.5*(M+vT),vT/eta+((0.5)*((beta.array().pow(2))/lambda.array()).sum())/sigmaE);
      tau=A;

      std::cout <<"tau" << tau<<"\n";

      blockNo=1;
      b=M/B;
      mu_f.setZero();

      VectorXd temp(M);
     // temp=(lambda*tau).cwiseInverse();
      temp=((tau*c2*lambda.array()).cwiseQuotient(tau*lambda.array()+c2)).cwiseInverse();
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
                   X.block(0,beginSegment,N,b).transpose()*(Y-mu_b-mu_f)/sigmaE,
                   xtX.block(beginSegment,beginSegment,b,b)/sigmaE,
                   temp.segment(beginSegment,b),1.0); //check which one is the correct expression (substitute 1.0 with sigmaE)

        mu_f+=X.block(0,beginSegment,N,b)*beta.block(beginSegment,0,b,1);
      }


      residues=mu_f;
      eta = inv_gamma_rate_rng(0.5+0.5*vT,(1.0/(pow(A,2))+vT/tau));
      v=(vL/(lambda).array()+1.0/phi.array()).unaryExpr(inv_gamma_functor<double>(vL));
      phi=(1.0/v.array()+1.0/chi.array()).unaryExpr(inv_gamma_functor<double>(1.0));
      chi=(1.0+1.0/phi.array()).unaryExpr(inv_gamma_functor<double>(1.0));
     // sigmaE=inv_scaled_chisq_rng(v0E+N,((Y-residues).squaredNorm()+v0E*s02E)/(v0E+N));
      sigmaE=inv_gamma_rate_rng( 0.5*(N+M),(Y-residues).squaredNorm()*0.5 + 0.5*(beta.array().pow(2)/(((tau*c2*lambda.array()).cwiseQuotient(tau*lambda.array()+c2)).cwiseInverse())).sum());
      //sigmaE=0.5;
      std::cout << sigmaE<<"\n";
      sum_beta_sqr= (1.0/N)*residues.squaredNorm() - pow(residues.mean(),2);
      if(iteration >= burn_in)
      {
        if(iteration % thinning==0){
          sample<< iteration,mu,beta,sigmaE,tau,lambda, sum_beta_sqr;
          q.enqueue(sample);
        }

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
MT=200
B=matrix(rnorm(M,sd=sqrt(0.5/MT)),ncol=1)
B[sample(1:M,M-MT),1]=0
  X <- matrix(rnorm(M*N), N, M); var(X[,1])
    G <- X%*%B; var(G)
      Y=X%*%B+rnorm(N,sd=sqrt((1-var(G)))); var(Y)
        Y=scale(Y)
        X=scale(X)
        vT=1
        vL=1
        A=0.001
       HorseshoePlus("./test2.csv",1, 2000,1000 ,1,X, Y,A,N,1-var(G),vL,vT,2)
        library(readr)
        tmp <- read_csv("./test2.csv")
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
        plot(tmp$sigmaE)
        plot(tmp$`beta[1]`)



        */

