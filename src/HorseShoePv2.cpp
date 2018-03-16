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
void HorseshoePlusv2(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,double A, double v0E, double s02E, double vL, double vT, int B,double c2) {
  int flag;
  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
  flag=0;
  int N(Y.size());
  int M(X.cols());
  VectorXd components(M);

  ////////////validate inputs

  if(max_iterations < burn_in || max_iterations<1 || burn_in<1) //validations related to mcmc burnin and iterations
  {
    std::cout<<"error: burn_in has to be a positive integer and smaller than the maximum number of iterations ";
    return;
  }



  Eigen::initParallel();
  Eigen::setNbThreads(10);
  double sum_beta_sqr;


#pragma omp parallel num_threads(2) shared(flag,q,M,N)
{
#pragma omp sections
{

  {

    //mean and residual variables
    double mu; // mean or intercept
    double sigmaG; //genetic variance
    double sigmaE; // residuals variance

    //component variables
    VectorXd lambda(M);
    VectorXd v(M);
    VectorXd phi(M);
    VectorXd chi(M);
    double tau;
    double eta;
    //linear model variables
    MatrixXd beta(M,1); // effect sizes
    VectorXd y_tilde(N); // variable containing the adjusted residuals to exclude the effects of a given marker
    VectorXd epsilon(N); // variable containing the residuals

    //sampler variables
    VectorXd sample(2*M+4+N); // varible containg a sambple of all variables in the model, M marker effects, M component assigned to markers, sigmaE, sigmaG, mu, iteration number and Explained variance
    std::vector<int> markerI;
    for (int i=0; i<M; ++i) {
      markerI.push_back(i);
    }


    int marker;
    double acum;


    y_tilde.setZero();

    beta.setZero();
    tau=beta_rng(1,1);

    mu=0;

    eta=inv_gamma_rate_rng(0.5,1/pow(A,2));
    eta=0.00001;
    std::cout<< "initial eta " << eta<<"\n";
    tau=(1.0/eta)*inv_gamma_rate_rng(0.5*vT,vT);

    tau=A;
    // tau=1/A;
    std::cout<< "initial tau " << tau<<"\n";

    v=(v.setOnes().array()).unaryExpr(inv_gamma_functor<double>(0));
    v.setOnes();
    //std::cout<< "initial v" << eta;
    lambda=v.unaryExpr(inv_gamma_functor_init_v<double>(vL));
    lambda.setOnes();




    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    epsilon= Y.array() - mu - (X*beta).array();
    sigmaE=epsilon.squaredNorm()/N*0.5;
    for(int iteration=0; iteration < max_iterations; iteration++){

      if(iteration>0)
        if( iteration % (int)std::ceil(max_iterations/10) ==0)
        {
          std::cout << "iteration: "<<iteration <<"\n";
          std::cout<< " tau " << tau<<"\n";
          std::cout<< " eta " << eta<<"\n";
          std::cout<< "sigmaE" << sigmaE<<"\n";
        }


      epsilon= epsilon.array()+mu;//  we substract previous value
      mu = norm_rng(epsilon.sum()/(double)N, sigmaE/(double)N); //update mu
      epsilon= epsilon.array()-mu;// we substract again now epsilon =Y-mu-X*beta


      std::random_shuffle(markerI.begin(), markerI.end());

      eta = inv_gamma_rate_rng(0.5+0.5*vT,(1.0/(A*A))+vT/tau);
      v=(vL/(lambda).array()+1.0).unaryExpr(inv_gamma_functor<double>(vL));
      for(int j=0; j < M; j++){

        marker= markerI[j];


        y_tilde= epsilon.array()+(X.col(marker)*beta(marker,0)).array();//now y_tilde= Y-mu-X*beta+ X.col(marker)*beta(marker)_old



       // std::cout<< muk;
        //we compute the denominator in the variance expression to save computations
        //denom=X.col(marker).squaredNorm()+(sigmaE/(tau*c2*lambda[marker]/(tau*lambda[marker]+c2)));
        //muk for the other components is computed according to equaitons
        //muk= (X.col(marker).cwiseProduct(y_tilde)).sum()/denom;
        //beta(marker,0)=norm_rng(muk,sigmaE/denom);
        beta(marker,0)=(X.col(marker).cwiseProduct(y_tilde)).sum()/(X.col(marker).squaredNorm()+(sigmaE/(tau*c2*lambda[marker]/(tau*lambda[marker]+c2))))+sqrt(sigmaE/(X.col(marker).squaredNorm()+(sigmaE/(tau*c2*lambda[marker]/(tau*lambda[marker]+c2)))))*norm_rng(0,1);



        epsilon=y_tilde-X.col(marker)*beta(marker,0);//now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new

      }

      lambda=(vL*v.cwiseInverse()+(0.5*beta.cwiseProduct(beta)*(1.0/tau))).unaryExpr(inv_gamma_functor<double>(vL));
    //  if(iteration==0)
      //  std::cout<< " lambda " << lambda<<"\n";
      tau= inv_gamma_rate_rng(0.5*(M+vT),vT/eta+((0.5)*((beta.array().pow(2))/lambda.array()).sum()));

      //tau=A;
     // c2=inv_gamma_rate_rng(0.5*vC+0.5*M,vC*sC*0.5+0.5*beta.squaredNorm());
    //  c2=sC;



      sigmaE=inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));

      if(iteration >= burn_in)
      {
        if(iteration % thinning == 0){
          sample<< iteration,mu,beta,sigmaE,sigmaG,lambda,epsilon;
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
  VectorXd sampleq(2*M+4+N);
  IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
  outFile<< "iteration,"<<"mu,";
  for(unsigned int i = 0; i < M; ++i){
    outFile << "beta[" << (i+1) << "],";

  }
  outFile<<"sigmaE,"<<"sigmaG,";
  for(unsigned int i = 0; i < M; ++i){
    outFile << "comp[" << (i+1) << "],";
  }
  for(unsigned int i = 0; i < N; ++i){
    outFile << "epsilon[" << (i+1) << "],";
  }
  outFile<<"\n";

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
B=matrix(rnorm(MT,sd=sqrt(0.5)),ncol=1) #marker effects, M marquers explain approx 50% of the variance
B[sample(1:MT,MT-M),1]=0 #we set MT-M marker effects to zero
#B=-abs(B)
X <- matrix(rnorm(MT*N), N, MT); var(X[,1])
G <- X%*%B; var(G)
Y=X%*%B+rnorm(N,sd=sqrt(0.4)); var(Y)
Y=Y
X=scale(X)
P=0.5 #prior probability of a marker being excluded from the model
sigma0=0.01# prior  variance of a zero mean gaussian prior over the mean mu NOT IMPLEMENTED
v0E=0.01 # degrees of freedom over the inv scaled chi square prior over residuals variance
s02E=0.01 #scale of the inv scaled chi square prior over residuals variance
v0G=0.01 #degrees of freedom of the inv bla bla prior over snp effects
s02G=0.01 # scale for the same
BayesRSamplerV2("./test2.csv",2, 5000, 2000,10,X, Y,sigma0,v0E,s02E,v0G,s02G,P)
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
hist(as.matrix(tmp[,grep("comp",names(tmp))])) #histogram of components, component 0= variance 0, component 1= variance 0.0001 and so long so forth
*/

