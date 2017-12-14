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

inline MatrixXd AtA(const MapMatd& A) {
  int n(A.cols());
  return MatrixXd(n,n).setZero().selfadjointView<Lower>()
                      .rankUpdate(A.adjoint());
}


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
void BayesRSampler(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,double sigma0, double v0E, double s02E, double v0G, double s02G, int B) {
 int flag;
  std::stringstream buffer;
  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
  flag=0;
  int N(Y.size());
  int M(X.cols());
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

  #pragma omp parallel num_threads(2) shared(flag,q,M,N)
{
    #pragma omp sections
    {

{

  double mu;
  double sigmaG;
  double sigmaE;
  int m0;
  std::random_device rd;
  std::mt19937 gen(rd());
  int b(M/B);
  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::normal_distribution<> norm(0,1);
  VectorXd priorPi(4);
  VectorXd pi(4);
  VectorXd cVa(4);
  VectorXd invsqrtcVa(4);
  VectorXd residues;
  VectorXd components(M);
  MatrixXd beta(M,1);
  MatrixXd xtX(M,M);
  MatrixXd xtY(M,1);
  MatrixXd xS(M,1);
  MatrixXd sum_beta_sqr(1,max_iterations);
  Map<MatrixXd> xM(X.data(),N,M);
  VectorXd v(4);
  MatrixXd mu_b(N,1);
  VectorXd ones(N);
  VectorXd mu_f(N);
  VectorXd sample(2*M+5);
  int beginSegment;
  int endSegment;
  int blockNo;


  /////end of declarations//////
  Eigen::initParallel();
  Eigen::setNbThreads(5);
  ones.setOnes();

  std::cout<<" computing XtX\n";
  std::chrono::high_resolution_clock::time_point startt= std::chrono::high_resolution_clock::now();
  xtX=AtA(xM);
  std::chrono::high_resolution_clock::time_point stopt= std::chrono::high_resolution_clock::now();
  auto durationt = std::chrono::duration_cast<std::chrono::seconds>( startt - stopt ).count();
  std::cout << "crossproduct was computed in: "<<durationt << "s\n";

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
  sigmaE=std::abs(norm_rng(0,1));
  sigmaG=std::abs(norm_rng(0,1));
  pi=dirichilet_rng(priorPi);
  components.unaryExpr(categorical_init<double>(priorPi));
  // residues=Y-mu*ones;


  std::cout<<"block size " << b <<"\n";
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  residues= X*beta;

  for(int iteration=0; iteration < max_iterations; iteration++){
    std::cout << "iteration: "<<iteration <<"\n";
    mu = norm_rng((1/(double)N)*residues.sum(), 1.0/(sigma0+(double)N/sigmaE));
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
    sigmaG=inv_scaled_chisq_rng(v0G+m0,((beta.array()).pow(2).sum()+v0G*s02G)/(v0G+m0));
    sigmaE=inv_scaled_chisq_rng(v0E+N,((((Y-residues).array()-mu).array().pow(2)).sum()+v0E*s02E)/(v0E+N));
    v(0)=priorPi[0]+(components.array()==cVa[0]).count();
    v(1)=priorPi[1]+(components.array()==cVa[1]).count();
    v(2)=priorPi[2]+(components.array()==cVa[2]).count();
    v(3)=priorPi[3]+(components.array()==cVa[3]).count();
    pi=dirichilet_rng(v);

    sum_beta_sqr(iteration)= (1.0/N)*mu_f.squaredNorm() - pow(mu_f.mean(),2);
      //buffer << iteration<<"\n";//<<"\t"<< mu <<"\t"<< beta.col(1).transpose()<<"\t"<< sigmaG <<"\t"<<sigmaE <<"\t"<< components.transpose()<< "\n";
      if(iteration >= burn_in)
      {
        sample<< iteration,mu,beta,sigmaE,sigmaG,components, sum_beta_sqr(iteration);
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
//
}

    }
}

}
//TODO: Why once I call BayesRsampler again after sourced, it doesnt write the file?


/*** R
M=100
N=2000
B=matrix(rnorm(M,sd=sqrt(0.5/M)),ncol=1)
  X <- matrix(rnorm(M*N), N, M); var(X[,1])
    G <- X%*%B; var(G)
      Y=X%*%B+rnorm(N,sd=sqrt(1-var(G))); var(Y)
      Y=scale(Y)
      X=scale(X)
       BayesRSampler("test2.csv",1, 30000, 29000,1,X, Y,0.01,0.01,0.01,0.01,0.01,2)
       library(readr)
       tmp <- read_csv("~/repo/ctggroup/BayesRRcpp/src/test2.csv")
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

