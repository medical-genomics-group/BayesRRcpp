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
void BayesRSamplerV2(std::string outputFile, int seed, int max_iterations, int burn_in, int thinning, Eigen::MatrixXd X, Eigen::VectorXd Y,double sigma0, double v0E, double s02E, double v0G, double s02G, double B) {
  int flag;
  moodycamel::ConcurrentQueue<Eigen::VectorXd> q;
  flag=0;
  int N(Y.size());
  int M(X.cols());
  VectorXd ones(N);
  VectorXd components(M);
  Map<MatrixXd> xM(X.data(),N,M);
  ////////////validate inputs
  if(B>1 || B<=0) /////////we validate the number of blocks
  {
    std::cout<<"error: prior is greater than 1 ";
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
    //return;
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
    VectorXd sample(2*M+5);
    std::vector<int> markerI;
    for (int i=0; i<M; ++i) {
      markerI.push_back(i);
    }

    Vector4d logL;
    Vector4d muk;
    Vector3d denom;
    int marker;
    double acum;

    priorPi[0]=B;



    priorPi.segment(1,3)=B*cVa.segment(1,3).segment(1,3).array()/cVa.segment(1,3).segment(1,3).sum();
    y_tilde.setZero();
    cVa[0] = 0;
    cVa[1] = 0.0001;
    cVa[2] = 0.001;
    cVa[3] = 0.01;

    cVaI[0] = 0;
    cVaI[1] = 1000;
    cVaI[2] = 100;
    cVaI[3] = 10;

    beta=beta.setRandom();
    //beta=(beta.array().abs() > 1e-6  ).select(beta, MatrixXd::Zero(M,1));
    mu=norm_rng(0,1);
    sigmaE=beta_rng(1,1);
    sigmaG=beta_rng(1,1);
    pi=priorPi;

    components.setZero();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    epsilon= Y.array() - mu - (X*beta).array();
    for(int iteration=0; iteration < max_iterations; iteration++){

      std::cout << "iteration: "<<iteration <<"\n";
      epsilon= epsilon.array()+mu;//  we substract previous value
      mu = norm_rng(epsilon.sum()/(double)N, sigmaE/(double)N); //update mu
      epsilon= epsilon.array()-mu;// we substract again now epsilon =Y-mu-X*beta


      std::random_shuffle(markerI.begin(), markerI.end());

      m0=0;
      v.setZero();
      for(int j=0; j < M; j++){

        marker= markerI[j];


        y_tilde= epsilon.array()+(X.col(marker)*beta(marker,0)).array();//now y_tilde= Y-mu-X*beta+ X.col(marker)*beta(marker)_old



        muk[0]=0.0;//muk for the zeroth component=0

       // std::cout<< muk;
        //we compute the denominator in the variance expression to save computations
        denom=X.col(marker).squaredNorm()+(sigmaE/sigmaG)*cVaI.segment(1,3).array();
        //muk for the other components is computed according to equaitons
        muk.segment(1,3)= (X.col(marker).cwiseProduct(y_tilde)).sum()/denom.array();



        logL= pi.array().log();//first component probabilities remain unchanged


        //for the other three components I think that this is equivalent as in the fortran code:
        //s(kk)=-0.5d0*(logdetV-(rhs*uhat/vare))+log_p(sidx,kk)

        /*
        logL.segment(1,3)=logL.segment(1,3).array() - 0.5*((double)N*log(sigmaE)+(((sigmaG/sigmaE)*(X.col(marker).squaredNorm())*cVa.segment(1,3).array() + 1).array().log())).abs()+
                  0.5*(y_tilde.squaredNorm() - muk.segment(1,3).array()*((X.col(marker).cwiseProduct(y_tilde)).squaredNorm()))/sigmaE;

         */

        //here we change also the probability of the first component
        /*
        logL=logL.array() - 0.5*((double)N*log(sigmaE)+(((sigmaG/sigmaE)*(X.col(marker).squaredNorm())*cVa.array() + 1).array().log())).abs()+
          0.5*(y_tilde.squaredNorm() - muk.array()*((X.col(marker).cwiseProduct(y_tilde)).squaredNorm()))/sigmaE;
         */

        // Here we reproduce the fortran code
        logL.segment(1,3)=logL.segment(1,3).array() - 0.5*((((sigmaG/sigmaE)*(X.col(marker).squaredNorm())*cVa.segment(1,3).array() + 1).array().log()))+
          0.5*( muk.segment(1,3).array()*((X.col(marker).cwiseProduct(y_tilde)).sum()))/sigmaE;

        double p(beta_rng(1,1));//I use beta(1,1) because I cant be bothered in using the std::random or create my own uniform distribution, I will change it later

        acum=pi(0);


        //uncomment this next bit if you want also to estimate the probability of the zeroth component
        //acum=1.0/((logL.array()-logL[0]).exp().sum());

        for(int k=0;k<4;k++){
          if(p<=acum){
            if(k==0){
              beta(marker,0)=0;
            }else{
              beta(marker,0)=norm_rng(muk[k],sigmaE/denom[k-1]);
            }
            v[k]+=1.0;
            components[marker]=k;
            break;
          }else{
            acum+=1.0/((logL.segment(1,3).array()-logL[k+1]).exp().sum());
          }
        }
       epsilon=y_tilde-X.col(marker)*beta(marker,0);//now epsilon contains Y-mu - X*beta+ X.col(marker)*beta(marker)_old- X.col(marker)*beta(marker)_new

      }

      m0=M-v[0];
      sigmaG=inv_scaled_chisq_rng(v0G+m0,(beta.squaredNorm()*m0+v0G*s02G)/(v0G+m0));


      //std::cout<< "sigmaG: "<<sigmaG<<"\n";
      //check if epsilon are the residues
      sigmaE=inv_scaled_chisq_rng(v0E+N,((epsilon).squaredNorm()+v0E*s02E)/(v0E+N));
      //std::cout<< "sigmaE: "<<sigmaE<<"\n";



      pi=dirichilet_rng(v+Vector4d::Ones());

      //std::cout<< "pi:"<<pi<<"\n";
      sum_beta_sqr= (1.0/N)*(epsilon.array()-Y.array()+mu).pow(2).sum() - pow((epsilon.array()-Y.array()+mu).mean(),2);
      //buffer << iteration<<"\n";//<<"\t"<< mu <<"\t"<< beta.col(1).transpose()<<"\t"<< sigmaG <<"\t"<<sigmaE <<"\t"<< components.transpose()<< "\n";
      if(iteration >= burn_in)
      {
        if(iteration % thinning == 0){
          sample<< iteration,mu,beta,sigmaE,sigmaG,components, sum_beta_sqr;
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
#B=-abs(B)
X <- matrix(rnorm(MT*N), N, MT); var(X[,1])
G <- X%*%B; var(G)
Y=X%*%B+rnorm(N,sd=sqrt(1-var(G))); var(Y)
Y=scale(Y)
X=scale(X)
P=0.5 #prior probability of a marker being excluded from the model
sigma0=0.01# prior  variance of a zero mean gaussian prior over the mean mu NOT IMPLEMENTED
v0E=0.01 # degrees of freedom over the inv scaled chi square prior over residuals variance
s02E=0.01 #scale of the inv scaled chi square prior over residuals variance
v0G=-2 #degrees of freedom of the inv bla bla prior over snp effects
s02G=-2 # scale for the same
BayesRSamplerV2("./test2.csv",2, 50000, 20000,10,X, Y,sigma0,v0E,s02E,v0G,s02G,P)
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

