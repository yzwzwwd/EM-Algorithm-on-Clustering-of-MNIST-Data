

##Create the image_loading function according to the MNIST image data storage charactteristics
load_image_file = function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f, integer() ,n=1, endian='big') #  Magic number
  ret$n = readBin(f,integer(),n=1,endian='big')
  nrow = readBin(f,integer(),n=1,endian='big')
  ncol = readBin(f,integer(),n=1,endian='big')
  x = readBin(f,integer(),n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

##Create the label_loading function according to the MNIST label data storage charactteristics
load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f, integer() ,n=1 ,endian='big') # Magic number
  n = readBin(f, integer(),n=1,size=4,endian='big')
  y = readBin(f, integer(),n=n,size=1,signed=F)
  close(f)
  y
}


##Loading the training data separately from image data and label data
train<-load_image_file('train-images.idx3-ubyte')
train$y<-load_label_file('train-labels.idx1-ubyte')
test<-load_image_file('t10k-images.idx3-ubyte')
test$y<-load_label_file('t10k-labels.idx1-ubyte')

##Extract those with label of c(0,1,2,3,4)
train01234<-list()
train01234$x<-train$x[train$y %in% c(0,1,2,3,4),]
train01234$y<-train$y[train$y %in% c(0,1,2,3,4)]
train01234$n<-length(train01234$y)
test01234<-list()
test01234$x<-test$x[test$y %in% c(0,1,2,3,4),]
test01234$y<-test$y[test$y %in% c(0,1,2,3,4)]
test01234$n<-length(test01234$y)

##Create a downsampling function which take the mean value of the original 2*2 pixel area  
##For example:now[1]=(old[1]+old[2]+old[29]+old[30])/4  
##This step could reduce the image size but lose some information
downsampling = function(im){
  ret = vector(length = 14*14)
  for(i in 1:14){
    for(j in 1:14){
      ret[14*(i-1)+j]=(im[28*2*(i-1)+(2*j-1)]+im[28*2*(i-1)+2*j]+im[28*(2*i-1)+2*(j-1)+1]+im[28*(2*i-1)+2*j])/4
    }
  }
  ret
}

##Apply the downsampling method to the image data
train01234$x<-t(apply(train01234$x,1,downsampling))
test01234$x<-t(apply(test01234$x,1,downsampling))

##Create a function to show the image by digit
show_digit<-function(im_arr, col=gray(12:1/12),size=14){
  im = matrix(im_arr, nrow=size)
  im<-t(apply(im,1,rev))
  image(1:size, 1:size,im,col=col)
}

par(mfrow=c(2,2))
for (i in 1:4){
  show_digit(train01234$x[i+2,]) 
}

##----------------------------------------------------------------------------------------------------------

##||(A) Spherical Models ||                                                       |
##-------------------------------------------------------------------
##First we divide each feature variable by its range--255 to avoid underflow problem
trainX<-train01234$x/255
testX<-test01234$x/255

GMM.spher<-function(X, mu, sigma, pi, tol, maxiter){
  n<-nrow(X)
  d<-ncol(X)
  likely<-numeric(0)
  likely[1]<-1
  ##In the multiplier inside exp operation, we need the sum of each row of the square of X
  ##We make this operation out of the iteration to speed up the whole process
  step1<-apply(X^2,1,sum)
  for(k in 1:maxiter){
    ##1.The quardratic term of x inside the exponent operator:
    qt<-(-1/2)*step1%*%t((1/sigma))
    ##2.The linear term of x inside the exponent operator:
    step2<-matrix(0,5,d)
    for (i in 1:5) {
      step2[i,]<-mu[i,]/sigma[i]
    }
    lt<-X%*%t(step2)
    ##3.The constant term inside the exponent operator:
    sumlinear<-apply(mu*step2,1,sum)
    cons<-(-1/2)*(sumlinear+d*log(sigma)) + log(pi) -1/2*d*log(2*base::pi)
    consmatrix<-matrix(cons,n,5,byrow = TRUE)
    ##Sum up to the exponential we want
    insideexp<-qt+lt+consmatrix
    ##Here we conduct a step for preparations for further operation: log sum trick  
    ##In further step, we need to use the sum of log(conditional probabilities) for each sample (each row in F matrix)  
    ##So here we use the log sum trick to avoid the underflow problem  
    ##we take out the largest of each row before exp operation
    A<-apply(insideexp,1,max)
    aftertrick<-matrix(0,n,5)
    for(i in 1:n){
      aftertrick[i,]<-insideexp[i,]-A[i]
    }
    ##Exp operation, each element equals to the Gaussian density times corresponding pi
    beforedivide <- exp(aftertrick)
    #We need each row's sums as the denominators
    denomi1<-apply(beforedivide,1,sum)
    F<-matrix(0,n,5)
    for (i in 1:n) {
      F[i,]<-beforedivide[i,]/denomi1[i]
    }
    ##Also, during each iteration, we need to calculate the log-likelihood value
    ##The max component for each row we extracted before is added back now
    likely[k+1]<-sum(log(denomi1))+sum(A)
    ##Break conditions
    if(abs(likely[k+1]-likely[k])/abs(likely[k])<tol){
      break
    }
    ##Maximization step
    ##For the iterated mu, according to the formula given in the note: 
    ##mu
    ##Nominator
    mu<-t(F)%*%X
    ##Denominator
    for (i in 1:5) {
      mu[i,]<-mu[i,]/apply(F,2,sum)[i]
    }
    ##For the iterated sigma
    ##We just need to interate out 5 numbers because it is spherical
    sigma<-drop(step1%*%F)-2*colSums(F*X%*%t(mu))+colSums(F)*rowSums(mu^2)
    sigma<-sigma/(apply(F,2,sum)*d)
    ##For the iterated pi, according to the formula given in the note:  
    ##New pj equals to sum of Fj among all samples and divide it by n
    ##Not dependent on the model type
    pi<-apply(F,2,sum)/n
    ##End of iterations
  }
  cluster<-apply(F, 1, which.max)

  return(list(mu = mu, sigma = sigma, pi = pi, log_likelihood=likely, cluster=cluster))
}

set.seed(5)
mu1<-matrix(runif(5*196),5,196)
sigma1<-runif(5)
pi1<-runif(5)
set.seed(50)
mu2<-matrix(runif(5*196),5,196)
sigma2<-runif(5)
pi2<-runif(5)
set.seed(500)
mu3<-matrix(runif(5*196),5,196)
sigma3<-runif(5)
pi3<-runif(5)


spher1<-GMM.spher(trainX, mu1, sigma1, pi1,  10^-4, 1000)
spher2<-GMM.spher(trainX, mu2, sigma2, pi2,  10^-4, 1000)
spher3<-GMM.spher(trainX, mu3, sigma3, pi3,  10^-4, 1000)
spher1$log_likelihood[length(spher1$log_likelihood)]
spher2$log_likelihood[length(spher2$log_likelihood)]
spher3$log_likelihood[length(spher3$log_likelihood)]
##By comparison, we choose the model with 3rd initialization
##Store the 3rd model's parameters for further validation  
spher_param<-list(mu=spher3$mu,sigma=spher3$sigma,pi=spher3$pi)
##In the training model, we get the cluster of each training sample according to  
##the maximum value of the final F matrix among each row.  
##However, the relationship between the clusters we assigned during the training model and the  
##true label for each image is ambiguous.  
##So we need to find out a map between the assigned clusters{1,2,3,4,5} and the true labels {0,1,2,3,4}
##We should decide the pattern based on the overlapping ratio (maximize the ratio)
spher_mappingmatrix<-matrix(0,5,5)
for (i in 1:5) {
  for (j in 1:5) {
    spher_mappingmatrix[i,j]=sum(spher3$cluster==i&train01234$y==(j-1))
  }
}
rownames(spher_mappingmatrix)<-c("cluster1","cluster2","cluster3","cluster4","cluster5")
colnames(spher_mappingmatrix)<-c("label0","label1","label2","label3","label4")
spher_mappingmatrix
##According to the matrix, we can see that the optimal mapping from clusters to labels should be:
spher_mapping<-c(0,1,4,2,3)
##Then we will use this mapping to calculate the error rate for the test set
##Use the train iterated parameters to calculate F matrix
GMM.spher.test<-function(X, mu, sigma, pi){
  n <- nrow(X)
  d <- ncol(X)
  ##In the multiplier inside exp operation, we need the sum of each row of the square of X
  ##We make this operation out of the iteration to speed up the whole process
  step1<-apply(X^2,1,sum)
  ##1.The quardratic term of x inside the exponent operator:
  qt<-(-1/2)*step1%*%t((1/sigma))
  ##2.The linear term of x inside the exponent operator:
  step2<-matrix(0,5,d)
  for (i in 1:5) {
    step2[i,]<-mu[i,]/sigma[i]
  }
  lt<-X%*%t(step2)
  ##3.The constant term inside the exponent operator:
  sumlinear<-apply(mu*step2,1,sum)
  cons<-(-1/2)*(sumlinear+d*log(sigma)) + log(pi) -1/2*d*log(2*base::pi)
  consmatrix<-matrix(cons,n,5,byrow = TRUE)
  ##Sum up to the exponential we want
  insideexp<-qt+lt+consmatrix
  ##Here we conduct a step for preparations for further operation: log sum trick  
  ##In further step, we need to use the sum of log(conditional probabilities) for each sample (each row in F matrix)  
  ##So here we use the log sum trick to avoid the underflow problem  
  ##we take out the largest of each row before exp operation
  A<-apply(insideexp,1,max)
  aftertrick<-matrix(0,n,5)
  for(i in 1:n){
    aftertrick[i,]<-insideexp[i,]-A[i]
  }
  ##Exp operation, each element equals to the Gaussian density times corresponding pi
  beforedivide <- exp(aftertrick)
  #We need each row's sums as the denominators
  denomi1<-apply(beforedivide,1,sum)
  F<-matrix(0,n,5)
  for (i in 1:n) {
    F[i,]<-beforedivide[i,]/denomi1[i]
  }
  return(F)
}
spher_test<-GMM.spher.test(testX, spher_param$mu, spher_param$sigma, spher_param$pi)
##Derive the test clusters according to the F matrix  
spher_testcluster<-apply(spher_test,1,which.max)  

spher_testmatrix<-matrix(0,5,5)
for (i in 1:5) {
  for (j in 1:5) {
    spher_testmatrix[i,j]=sum(spher_testcluster==i&test01234$y==(j-1))
  }
}
rownames(spher_testmatrix)<-c("predict0","predict1","predict4","predict2","predict3")
colnames(spher_testmatrix)<-c("true0","true1","true2","true3","true4")
spher_testmatrix

spher_correctassig<-numeric(0)
for (i in 1:5) {
  spher_correctassig[i]=sum(spher_testcluster==i&test01234$y==spher_mapping[i])
}
##Prediction Error Rate
1-sum(spher_correctassig)/length(test01234$y)
##----------------------------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------
##||(B) Diagonal Models ||                                                       |
##-------------------------------------------------------------------
##Diagonal covariance
GMM.diag<-function(X, mu, sigma, pi, step, tol, maxiter){
  n<-nrow(X)
  d<-ncol(X)
  likely<-numeric(0)
  likely[1]<-1
  for(k in 1:maxiter){
    ##1.The quardratic term of x inside the exponent operator:
    qt<-(-1/2)*X^2%*%t((1/sigma))
    ##2.The linear term of x inside the exponent operator:
    ##mu and sigma share the same shape
    lt<-X%*%t(mu/sigma)
    ##3.The constant term inside the exponent operator:
    step2<-mu^2/sigma
    cons<-(-1/2)*apply(step2+log(sigma),1,sum)-(1/2)*d*log(2*base::pi)+log(pi)
    consmatrix<-matrix(cons,n,5,byrow = TRUE)
    ##Sum up to the exponential we want
    insideexp<-qt+lt+consmatrix
    ##Calculate F
    ##Here we conduct a step for preparations for further operation: log sum trick  
    ##In further step, we need to use the sum of log(conditional probabilities) for each sample (each row in F matrix)  
    ##So here we use the log sum trick to avoid the underflow problem  
    ##we take out the largest of each row before exp operation
    A<-apply(insideexp,1,max)
    aftertrick<-matrix(0,n,5)
    for(i in 1:n){
      aftertrick[i,]<-insideexp[i,]-A[i]
    }
    ##Exp operation, each element equals to the Gaussian density times corresponding pi
    beforedivide <- exp(aftertrick)
    #We need each row's sums as the denominators
    denomi2<-apply(beforedivide,1,sum)
    F<-matrix(0,n,5)
    for (i in 1:n) {
      F[i,]<-beforedivide[i,]/denomi2[i]
    }
    ##Also, during each iteration, we need to calculate the log-likelihood value
    ##The max component for each row we extracted before is added back now
    likely[k+1]<-sum(log(denomi2))+sum(A)
    ##Break conditions
    if(abs(likely[k+1]-likely[k])/abs(likely[k])<tol){
      break
    }
    ##Calculate the final parameters value which will be used for prediction for test data set  
    ##The formula for mu and pi are the same which are not dependent on the model
    ##mu
    ##Nominator
    mu<-t(F)%*%X
    step5<-apply(F,2,sum)
    ##Denominator
    for (i in 1:5) {
      mu[i,]<-mu[i,]/step5[i]
    }
    ##pi
    pi<-step5/n
    ##sigma
    sig_cons<-matrix(0,5,d)
    for (i in 1:d) {
      sig_cons[,i]=(mu^2)[,i]*step5
    }
    sigma<-t(F)%*%X^2-2*mu*t(F)%*%X+sig_cons
    for (i in 1:d) {
      sigma[,i]=sigma[,i]/step5
    }
    sigma<-sigma+step
  }
  
  cluster<-apply(F, 1, which.max)
  return(list(pi = pi,mu = mu, sigma = sigma,log_likelihood=likely, cluster = cluster) )
}

##Assume 3 different random initializations  
set.seed(5)
mu1<-matrix(runif(ncol(trainX)*5), 5, ncol(trainX))
sigma1<-matrix(runif(5*ncol(trainX)), 5, ncol(trainX))
pi1<-rep(1/5, 5)
set.seed(50)
mu2<-matrix(runif(ncol(trainX)*5), 5, ncol(trainX))
sigma2<-matrix(runif(5*ncol(trainX)), 5, ncol(trainX))
pi2<-rep(1/5, 5)
set.seed(500)
mu3<-matrix(runif(ncol(trainX)*5), 5, ncol(trainX))
sigma3<-matrix(runif(5*ncol(trainX)), 5, ncol(trainX))
pi3<-rep(1/5, 5)

diag1<-GMM.diag(trainX, mu1, sigma1, pi1, 0.05, 10^-6, 1000)
diag2<-GMM.diag(trainX, mu2, sigma2, pi2, 0.05, 10^-6, 1000)
diag3<-GMM.diag(trainX, mu3, sigma3, pi3, 0.05, 10^-6, 1000)
diag1$log_likelihood[length(diag1$log_likelihood)]
diag2$log_likelihood[length(diag2$log_likelihood)]
diag3$log_likelihood[length(diag3$log_likelihood)]
##We can see that the reuslts are quite close to each other  
##We choose the parameters from diag model 3 because it has the maximum log likelihood value  
diag_param<-list(mu=diag3$mu,sigma=diag3$sigma,pi=diag3$pi)
##In the training model, we get the cluster of each training sample according to  
##the maximum value of the final F matrix among each row.  
##However, the relationship between the clusters we assigned during the training model and the  
##true label for each image is ambiguous.  
##So we need to find out a map between the assigned clusters{1,2,3,4,5} and the true labels {0,1,2,3,4}
##We should decide the pattern based on the overlapping ratio (maximize the ratio)
diag_mappingmatrix<-matrix(0,5,5)
for (i in 1:5) {
  for (j in 1:5) {
    diag_mappingmatrix[i,j]=sum(diag3$cluster==i&train01234$y==(j-1))
  }
}
rownames(diag_mappingmatrix)<-c("cluster1","cluster2","cluster3","cluster4","cluster5")
colnames(diag_mappingmatrix)<-c("label0","label1","label2","label3","label4")
diag_mappingmatrix
##According to the matrix, we can see that the optimal mapping from clusters to labels should be:
diag_mapping<-c(1,2,3,0,4) #This mapping seems much better than that of spherical one, which may indicate a much better prediction performance
##Then we will use this mapping to calculate the error rate for the test set
##Use the train iterated parameters to calculate F matrix
GMM.diag.test<-function(X, mu, sigma, pi){
  n <- nrow(X)
  d <- ncol(X)
  ##1.The quardratic term of x inside the exponent operator:
  qt<-(-1/2)*X^2%*%t((1/sigma))
  ##2.The linear term of x inside the exponent operator:
  ##mu and sigma share the same shape
  lt<-X%*%t(mu/sigma)
  ##3.The constant term inside the exponent operator:
  step2<-mu^2/sigma
  cons<-(-1/2)*apply(step2+log(sigma),1,sum)-(1/2)*d*log(2*base::pi)+log(pi)
  consmatrix<-matrix(cons,n,5,byrow = TRUE)
  ##Sum up to the exponential we want
  insideexp<-qt+lt+consmatrix
  ##Calculate F
  ##Here we conduct a step for preparations for further operation: log sum trick  
  ##In further step, we need to use the sum of log(conditional probabilities) for each sample (each row in F matrix)  
  ##So here we use the log sum trick to avoid the underflow problem  
  ##we take out the largest of each row before exp operation
  A<-apply(insideexp,1,max)
  aftertrick<-matrix(0,n,5)
  for(i in 1:n){
    aftertrick[i,]<-insideexp[i,]-A[i]
  }
  ##Exp operation, each element equals to the Gaussian density times corresponding pi
  beforedivide <- exp(aftertrick)
  #We need each row's sums as the denominators
  denomi2<-apply(beforedivide,1,sum)
  F<-matrix(0,n,5)
  for (i in 1:n) {
    F[i,]<-beforedivide[i,]/denomi2[i]
  }
  return(F)
}
##fit the model with test data and return the F matrix
diag_test<-GMM.diag.test(testX, diag_param$mu, diag_param$sigma, diag_param$pi)
##Derive the test clusters according to the F matrix  
diag_testcluster<-apply(diag_test,1,which.max)

diag_testmatrix<-matrix(0,5,5)
for (i in 1:5) {
  for (j in 1:5) {
    diag_testmatrix[i,j]=sum(diag_testcluster==i&test01234$y==(j-1))
  }
}
rownames(diag_testmatrix)<-c("predict1","predict2","predict3","predict0","predict4")
colnames(diag_testmatrix)<-c("true0","true1","true2","true3","true4")
diag_testmatrix

diag_correctassig<-numeric(0)
for (i in 1:5) {
  diag_correctassig[i]=sum(diag_testcluster==i&test01234$y==diag_mapping[i])
}
##Prediction Error Rate
1-sum(diag_correctassig)/length(test01234$y)
