## ------------------------------------------------------------------------
source('./sbc.R')

## ----message=F,error=F,warning=F,results=F-------------------------------

M=200 #non zero marker effects
N=2000 #observations
MT=5000 #number of markers
B=matrix(rnorm(MT,sd=sqrt(0.5/M)),ncol=1) #marker effects, M marquers explain approx 50% of the variance
B[sample(1:MT,MT-M),1]=0 #we set MT-M marker effects to zero
#B=-abs(B)
X <- matrix(rnorm(MT*N), N, MT); var(X[,1])
G <- scale(X)%*%B;
var(G)
Y=G

Covariates <- matrix(rnorm(N*2,sd = 1),ncol=2)
C <- c(1/(2*sqrt(5)),-1/(2*sqrt(5)))
Y = Y+ scale(Covariates) %*% C
var(scale(Covariates) %*% C)
e = rnorm(N,sd=sqrt(0.4))
Y = Y + e
var(rnorm(N,sd=sqrt(0.4)))

#Y=scale(Y)
X=scale(X)
iter=20000
burnin =10000
thin=10
Rcpp::sourceCpp('../src/BayesRv2.cpp')

## ----eval=F--------------------------------------------------------------
## source('./sbc.R')
##
 sbc.result.sB <- sbc(scale(X),scale(Covariates),num.params = 0,sbc.sweeps = 100,posterior.draws = 1000,init.thin = 5,max.thin = 100, target.neff = 900,FUN = BayesRSamplerV2sigmaB)
save(list='sbc.result.sB', file = 'sbc.result.sB.RData')
## hist(sbc.result$rank,breaks=100)

## ----eval=F--------------------------------------------------------------
## source('./sbc.R')
##
 sbc.result.sG <- sbc(scale(X),scale(Covariates),num.params = 100,sbc.sweeps = 1,posterior.draws = 1000,init.thin = 5,max.thin = 100, target.neff = 900,FUN = BayesRSamplerV2)
save(list='sbc.result.sG', file = 'sbc.result.sG.RData')
##  C <- as.matrix(data.table::fread('sbc1.csv'))
##   params <- param.family(C,"sigmaG|sigmaE|beta|gamma")
## hist(sbc.result$rank,breaks=100)

