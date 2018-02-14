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

inline double bernoulli_rng(double probs0,double probs1,double cats0,double cats1){
  double p;
  p=R::runif(0,1);
  if(p<= probs0/(probs0+probs1))
    return cats0;
  else
    return cats1;
}

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
  inv_gamma_functor(const Scalar& vd,const Scalar& vc):m_a(vd),m_b(vc){}

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rate_rng(0.5+m_a,m_b+x); }
  Scalar  m_a;
  Scalar m_b;
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

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rng(0.5,x); }

};

template<typename Scalar>
struct inv_gamma_functor_init_v
{
  inv_gamma_functor_init_v(const Scalar& vd,const Scalar& vc):m_a(vd),m_b(vc){}

  const Scalar operator()(const Scalar& x) const{ return inv_gamma_rate_rng(0.5+m_a,m_b+x); }
  Scalar  m_a;
  Scalar m_b;
};

template<typename Scalar>
struct bernoulli_functor
{
  bernoulli_functor(const Scalar& cats0,const Scalar& cats1, const Scalar& p ):m_a(cats0),m_b(cats1),w(p){}
  inline const Scalar operator()(const Scalar& x) const{
    return bernoulli_rng((1-w)*exp(x/m_a)/sqrt(m_a),(w)*exp(x/m_b)/sqrt(m_b),m_a,m_b); }
  Scalar  m_a;
  Scalar  m_b;
  Scalar w;
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
void spikeSlab(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,Eigen::VectorXd cats, double v0E, double s02E, double v0L, double s02L, int B,double a, double b) {
  int flag;
  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
  flag=0;
  int N(Y.size());
  int M(X.cols());
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
  if( v0E < 0 || s02E < 0 || v0L < 0||  s02L < 0 )//validations related to hyperparameters
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
    VectorXd components(M);
    VectorXd residues(M);
    MatrixXd beta(M,1);
    double w;

    double sum_beta_sqr;

    VectorXd sample(2*M+5);

    int beginSegment;
    int endSegment;
    int blockNo;



    //beta=(xtX).colPivHouseholderQr().solve(X.transpose()*Y);
    beta.setRandom();
    mu=norm_rng(0,1);
    sigmaE=0.5;
    std::cout<< "initial SigmaE " << sigmaE<<"\n";
    lambda.setConstant(1.0/(0.1)*M);
    components.setOnes();

    w=0.5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    residues= X*beta;

    for(int iteration=0; iteration < max_iterations; iteration++){
      //   std::cout <<beta.col(0) <<"\n";
      if(iteration>0)
        if( iteration % (int)std::ceil(max_iterations/10) ==0){
          std::cout << "iteration: "<<iteration <<"\n";
          std::cout << "sigmaE: " << sigmaE<<"\n";
        }


      blockNo=1;
      b=M/B;
      mu_f.setZero();

      VectorXd temp(M);
      temp=(components.cwiseProduct(lambda)).cwiseInverse();

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
                   sigmaE*temp.segment(beginSegment,b),sqrt(sigmaE)); //check which one is the correct expression (substitute 1.0 with sigmaE)

        mu_f+=X.block(0,beginSegment,N,b)*beta.block(beginSegment,0,b,1);
      }
      residues=mu_f;
      components=(-0.5*beta.col(0).array().pow(2)/lambda.array()).unaryExpr(bernoulli_functor<double>(cats[0],cats[1],w));
      //components.setOnes();
      lambda=(0.5*beta.col(0).array().pow(2)/components.array()).unaryExpr(inv_gamma_functor<double>(v0L*0.5,v0L*s02L*0.5));

      w=beta_rng(a+(components.array()==cats[1]).count(),b+(components.array()==cats[0]).count());
      sigmaE=inv_gamma_rng(0.5*v0E+0.5*N,0.5*v0E*s02E + 0.5*(Y-residues).squaredNorm());

      sum_beta_sqr= (1.0/N)*residues.squaredNorm() - pow(residues.mean(),2);
      if(iteration >= burn_in)
      {
        if(iteration % thinning ==0){
          sample<< iteration,mu,beta,sigmaE,w,lambda, sum_beta_sqr;
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
M=200
N=2000
MT=3000
B=matrix(rnorm(MT,sd=sqrt(0.5/M)),ncol=1)
  B[sample(1:MT,MT-M),1]=0
X <- matrix(rnorm(M*N), N, MT); var(X[,1])
  G <- X%*%B; var(G)
    Y=X%*%B+rnorm(N,sd=sqrt(1-var(G))); var(Y)
      Y=Y
      X=scale(X)
      vL=1
    sL=0.1
    A=c(0.000005,1)
      spikeSlab("./test4.csv",1, 50000,30000 ,10,X, Y,A,1,1-var(G),vL,sL,100)
      library(readr)

      tmp <- read_csv("./test4.csv")

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


      plot(tmp$sigmaE)
      plot(tmp$`beta[1]`)


      */
