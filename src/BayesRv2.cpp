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
void BayesRSamplerV2(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,double sigma0, double v0E, double s02E, double v0G, double s02G, int B) {
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
  //xtX=AtA(xM);
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
    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    Vector4d priorPi;
    Vector4d pi;
    Vector4d cVa;
    MatrixXd beta(M,1);


    Vector4d v;
    Vector4d cVaI;

    VectorXd y_tilde(N);
    VectorXd epsilon(N);
   // VectorXi markerI(M);
    VectorXd sample(M+5);
    std::vector<int> markerI;
    for (int i=0; i<M; ++i) {
      markerI.push_back(i);
    }

    Vector4d logL;
    Vector4d muk;
    Vector3d denom;
    int marker;
    double num;
    double acum;

    y_tilde.setZero();
    priorPi.setOnes();
    priorPi*=0.25;
    cVa[0] = 0;
    cVa[1] = 0.0001;
    cVa[2] = 0.001;
    cVa[3] = 0.01;

    cVaI.setZero();
    cVaI.segment(1,3)=cVa.segment(1,3).cwiseInverse();
    beta=beta.setRandom();
    beta=(beta.array() > 1e-6  ).select(beta, MatrixXd::Zero(M,1));
    mu=norm_rng(0,1);
    sigmaE=0.05*beta_rng(1,1);
    sigmaG=0.05*beta_rng(1,1);
    pi=dirichilet_rng(priorPi);


    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    epsilon= Y.array() - mu - (X*beta).array();
    for(int iteration=0; iteration < max_iterations; iteration++){

      std::cout << "iteration: "<<iteration <<"\n";
      epsilon= epsilon.array()+mu;
      mu = norm_rng(epsilon.sum()/(double)N, sigmaE/(double)N);


      epsilon= epsilon.array()-mu;


      std::random_shuffle(markerI.begin(), markerI.end());

      m0=0;
      v=priorPi;
      for(int j=0; j < M; j++){

        marker= markerI[j];


        y_tilde= epsilon.array()+(X.col(marker)*beta(marker,0)).array();



        muk[0]=0.0;
        num=X.col(marker).cwiseProduct(y_tilde).array().sum();

        denom=X.col(marker).squaredNorm()+(sigmaE/sigmaG)*cVaI.segment(1,3).array();

        muk.segment(1,3)= num/denom.array();

        logL= pi.array().log() -
          0.5*((double)N*log(sigmaE)+(((sigmaG/sigmaE)*(X.col(marker).squaredNorm())*cVa.array() + 1).array().log()).abs())-
                  0.5*(y_tilde.squaredNorm()-muk.array()*X.col(marker).cwiseProduct(y_tilde).squaredNorm()).array()/sigmaE;


        double p(beta_rng(1,1));
        acum=1.0/((logL.array()-logL[0]).exp().sum());

        for(int k=0;k<4;k++){
          if(p<=acum){
            if(k==0){
              beta(marker,0)=0;
            }else{
              beta(marker,0)=norm_rng(muk[k],sigmaE/denom[k-1]);
            }
            v[k]+=1.0;
            break;
          }else{
            acum+=1.0/((logL.array()-logL[k+1]).exp().sum());
          }
        }
       epsilon=y_tilde-X.col(marker)*beta(marker,0);

      }

      m0=M-v[0]-priorPi[0];
      sigmaG=inv_scaled_chisq_rng(v0G+m0,(beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));


      //std::cout<< "sigmaG: "<<sigmaG<<"\n";
      //check if epsilon are the residues
      sigmaE=inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));
      //std::cout<< "sigmaE: "<<sigmaE<<"\n";



      pi=dirichilet_rng(v);

      //std::cout<< "pi:"<<pi<<"\n";
      sum_beta_sqr= (1.0/N)*epsilon.squaredNorm() - pow(epsilon.mean(),2);
      //buffer << iteration<<"\n";//<<"\t"<< mu <<"\t"<< beta.col(1).transpose()<<"\t"<< sigmaG <<"\t"<<sigmaE <<"\t"<< components.transpose()<< "\n";
      if(iteration >= burn_in)
      {
        sample<< iteration,mu,beta,sigmaE,sigmaG, sum_beta_sqr;
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
M=200 #non zero marker effects
N=2000 #observations
MT=2000 #number of markers
B=matrix(rnorm(MT,sd=sqrt(0.5/M)),ncol=1) #marker effects, M marquers explain approx 50% of the variance
B[sample(1:MT,MT-M),1]=0 #we set MT-M marker effects to zero
X <- matrix(rnorm(MT*N), N, MT); var(X[,1])
G <- X%*%B; var(G)
Y=X%*%B+rnorm(N,sd=sqrt(1-var(G))); var(Y)
Y=scale(Y)
X=scale(X)
BayesRSamplerV2("./test2.csv",2, 10000, 9000,1,X, Y,0.01,0.01,0.01,0.01,0.01,100)
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
plot(tmp$mu)
plot(tmp$sigmaE)
plot(tmp$sigmaG)

*/

