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

inline MatrixXd AtA(const MapMatd& A) {
  int n(A.cols());
  return MatrixXd(n,n).setZero().selfadjointView<Lower>()
                      .rankUpdate(A.adjoint());
}
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
 * B- integer lower than number of columns of X, number of blocks in which the effects beta will be conditiionally divided as p(beta_B|beta_\B,.)
 */
// [[Rcpp::export]]
void BayesRSampler(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,double sigma0, double v0E, double s02E, double v0G, double s02G, int B) {
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
  if(sigma0 < 0 || v0E < 0 || s02E < 0 || v0G < 0||  s02G < 0 )//validations related to hyperparameters
  {
    std::cout<<"error: hyper parameters have to be positive";
    return;
  }
  /////end of declarations//////
  ones.setOnes();


  Eigen::initParallel();
  Eigen::setNbThreads(10);
  double sum_beta_sqr;
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

      double mu;
      double sigmaG;
      double sigmaE;
      int m0;
      int b(M/B);
      // values near the mean are the most likely
      // standard deviation affects the dispersion of generated values from the mean
      Vector4d priorPi;
      Vector4d pi;
      Vector4d cVa;
      VectorXd residues;
      VectorXd components(M);
      MatrixXd beta(M,1);


      Vector4d v;
      MatrixXd mu_b(N,1);

      VectorXd mu_f(N);

      VectorXd sample(2*M+5);
      int beginSegment;
      int endSegment;
      int blockNo;


      priorPi.setOnes();
      priorPi*=0.25;
      cVa[0] = 0;
      cVa[1] = 0.0001;
      cVa[2] = 0.001;
      cVa[3] = 0.01;

      beta.setRandom();
      mu=norm_rng(0,1);
      sigmaE=std::abs(norm_rng(0,1));
      sigmaG=std::abs(norm_rng(0,1));
      pi=dirichilet_rng(priorPi);
      components.unaryExpr(categorical_init<double>(priorPi));


      std::cout<<"block size " << b <<"\n";
      std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
      residues= X*beta;
      for(int iteration=0; iteration < max_iterations; iteration++){

        std::cout << "iteration: "<<iteration <<"\n";

        mu = norm_rng((1/(double)N)*residues.sum(), 1.0/(1.0/sigma0+(double)N/sigmaE));
        components= beta.unaryExpr(categorical_functor<double>(pi,sigmaG));
        mu_f.setZero();

        blockNo=1;
        b=M/B;
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
                 X.block(0,beginSegment,N,b).transpose()*(Y-mu*ones-mu_b-mu_f),
                 xtX.block(beginSegment,beginSegment,b,b),
                 sigmaG*components.segment(beginSegment,b));

          beta.block(beginSegment,0,b,1) = (components.block(beginSegment,0,b,1).array() > 1e-10 ).select(beta.block(beginSegment,0,b,1), MatrixXd::Zero(b,1));
          mu_f+=X.block(0,beginSegment,N,b)*beta.block(beginSegment,0,b,1);
      }

      residues=mu_f;
      m0=(components.array()>0).count();
      sigmaG=inv_scaled_chisq_rng(v0G+m0,((beta.array()).pow(2).sum()*m0+v0G*s02G)/(v0G+m0));
      sigmaE=inv_scaled_chisq_rng(v0E+N,((((Y-residues).array()-mu).array().pow(2)).sum()+v0E*s02E)/(v0E+N));
      v(0)=priorPi[0]+(components.array()==cVa[0]).count();
      v(1)=priorPi[1]+(components.array()==cVa[1]).count();
      v(2)=priorPi[2]+(components.array()==cVa[2]).count();
      v(3)=priorPi[3]+(components.array()==cVa[3]).count();
      pi=dirichilet_rng(v);

      sum_beta_sqr= (1.0/N)*mu_f.squaredNorm() - pow(mu_f.mean(),2);
      //buffer << iteration<<"\n";//<<"\t"<< mu <<"\t"<< beta.col(1).transpose()<<"\t"<< sigmaG <<"\t"<<sigmaE <<"\t"<< components.transpose()<< "\n";
      if(iteration >= burn_in)
      {
        sample<< iteration,mu,beta,sigmaE,sigmaG,components, sum_beta_sqr;
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
    outFile<<"sigmaE,"<<"sigmaG,";
    for(unsigned int i = 0; i < M; ++i){
      outFile << "comp[" << (i+1) << "],";
    }
    outFile<<"EV\n";
    while(!flag ){
      if(q.try_dequeue(sampleq))
        outFile<< sampleq.transpose().format(CommaInitFmt) << "\n";
    }
  }

  }
}

}


/*** R
M=100
N=2000
B=matrix(rnorm(M,sd=sqrt(0.5/M)),ncol=1)
  X <- matrix(rnorm(M*N), N, M); var(X[,1])
    G <- X%*%B; var(G)
      Y=X%*%B+rnorm(N,sd=sqrt(1-var(G))); var(Y)
     # Y=scale(Y)
    #  X=scale(X)
       BayesRSampler("./test2.csv",1, 30000, 29000,1,X, Y,0.01,0.01,0.01,0.01,0.01,2)
       library(readr)
       tmp <- read_csv("./test2.csv")
        #names(tmp)
        #plot(tmp$sigmaG); mean(tmp$sigmaG)
       plot(B,colMeans(tmp[,grep("beta",names(tmp))]))
        lines(B,B)
        abline(h=0)
         var(G)
        mean(tmp$EV)
         1-var(G)
         mean(tmp$sigmaE)

  */

