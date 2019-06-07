library(parallel)
library(foreach)
library(doParallel)
library(coda)
param.family <- function( chain.matrix, family){
  if(is.character(family))
    as.mcmc(chain.matrix[,grep(family,colnames(chain.matrix))])
  else
    stop("error, family has to be character")
}
Rcpp::sourceCpp('../src/distributions.cpp')
sbc <- function(X,covariates,num.params,sbc.sweeps,init.thin,posterior.draws,max.thin,target.neff,FUN){
  #ranks <- matrix(nrow = sbc.sweeps, ncol = num.params)
  thins <- rep(NA, sbc.sweeps)
  cl <- parallel::makeForkCluster(20)
  registerDoParallel(cl)
  ranks <- foreach( n =  1:sbc.sweeps,.combine=rbind) %dopar% {
    n_eff <- 0
    thin <- init.thin
    b<- rep (0,ncol(X))
    while(TRUE){
      sigma0=0.01# prior  variance of a zero mean gaussian prior over the mean mu NOT IMPLEMENTED YET
      v0E= 4 # degrees of freedom over the inv scaled chi square prior over residuals variance
      s02E = 0.5 #scale of the inv scaled chi square prior over residuals variance
      v0G = 4 #degrees of freedom of the inv bla bla prior over snp effects
      s02G = 0.5 # scale for the samecva
      cva=matrix(c(0.0001,0.001,0.01),nrow = 1) #components variance

      sigmaE <- inv_scaled_chisq_rng(v0E,s02E)
      print(sigmaE)
      sigmaG <- inv_scaled_chisq_rng(v0G,s02G)
      print(sigmaG)
      #sigmaF <- geoR::rinvchisq(1,v0E,s02E)
      sigmaF <- 10
      print(sigmaF)
      pi <- dirichilet_rng(alpha=rep(1,ncol(cva)+1) )
      print(length(pi))
      b <- sapply(b,FUN = function(x){ comp <- sample(c(0,cva[1,]),size = 1,replace = F,prob = pi)

      ifelse(comp==0,0,rnorm(1,sd= sqrt(sigmaG*comp)))
      })
      gamma <- rnorm(ncol(covariates),sqrt(sigmaF))
      g <- scale(X) %*% b;
      e <- rnorm(N,0, sqrt(sigmaE)) #residuals variance
      m <- scale(covariates) %*% gamma
      y = g + e + m
      sim.name <- paste("./sbc",paste( as.character(n), ".csv",sep=""),sep="")
      FUN( sim.name, 2, posterior.draws*2*thin, posterior.draws*thin ,thin,scale(X), Y, sigma0, v0E, s02E, v0G, s02G, cva, covariates)
      C <- as.matrix(data.table::fread(sim.name))
      if(!any(is.na(C))){
        chains.sigma <- mcmc.list( param.family( C, "sigma") )
        n_eff <- mean( coda::effectiveSize( chains.sigma ) )
        if(n_eff >= target.neff || (2*thin) > max.thin) break;
        print("effective sample size:")
        print(n_eff)
        thin <- 2*thin
        print("increasing thinning")
      }
      else{
        print("invalid chain")
      }
    }
    thins[n] <- thin
    ##for now without the pi
    params <- param.family(C,"sigmaG|sigmaE|beta|gamma")

    param.vector <- c(b, sigmaE, sigmaG, gamma)
    #print(param.vector)
    #print(dim(params))
    ranks.n <- rep(NA,length(param.vector))
    if(!any(is.na(params)))
      ranks.n <-  1 + rowSums(apply(params,MARGIN = 1, function(x){ x <  param.vector}))
    #print(length(ranks.n))
    #print(dim(ranks))
    #print(ranks.n)
    ranks.n
  }
  parallel::stopCluster(cl)
  list(rank =ranks,thin = thins)
}

