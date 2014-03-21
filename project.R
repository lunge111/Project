setwd("/media/Windows8_OS/14winter/250/project/")
train=read.csv("train.csv")
test=read.csv("test.csv")
library(parallel)
i=sample(1:42000,10000)
training=train[i,]
test=train[-i,]
j=sample(1:32000,10000)
test=test[j,]
#subset into 4 group
one=subset(training,label==1)
eight=subset(training,label==8)
three=subset(training,label==3)
seven=subset(training,label==7)
one$label=1
three$label=2
seven$label=3
eight$label=4
head(one)
training=rbind(one,three,seven,eight)
testing=subset(test,test$label%in%c(1,3,7,8))
testing[testing$label==3,]$label=2
testing[testing$label==7,]$label=3
testing[testing$label==8,]$label=4

##save n(n-1)/2 pairs to a list, they are:
#[1] 1 2
#[1] 1 3
#[1] 1 4
#[1] 2 3
#[1] 2 4
#[1] 3 4
pair=list();k=1
for(i in 1:3)
  for(j in (i+1):4){
    pair[[k]]=c(i,j);k=k+1;print(c(i,j))}
cl <- makeCluster(4,"FORK")
v=clusterSplit(cl,pair)


##find k nearest neighbor
findknn<-function(matr,k){
  #given  a distance matrix matr, find k-nn
  a=apply(matr,1,function(x) data.frame(num=order(x)[1:k],dis=x[order(x)[1:k]]))
  a
}
KNN<-function(matr=NULL,testdat,traindat,k,
              trainlabel,dismethod=c("euclidean","manhattan","mahalanobis")){
  require(StatMatch)
  if(is.null(matr)){
    if(dismethod=="mahalanobis")
      matr=mahalanobis.dist(testdat,traindat)
    if(dismethod=="euclidean")
      matr=as.matrix(dist(rbind(testdat,traindat)))[1:nrow(testdat),
                                                   -(1:nrow(testdat))]
    if(dismethod=="manhattan")
      matr=as.matrix(dist(rbind(testdat,traindat),method="manhattan"))[1:nrow(testdat),
                                                                       (nrow(testdat)+1):(nrow(testdat)+nrow(traindat))]
  }
  knn=findknn(matr,k)
  sapply(knn, function(x,trainlabel) {count=trainlabel[x$num]
                                      as.integer(names(sort(tapply(1/x$dis,count,sum),decreasing=T))[1])}
         ,trainlabel=trainlabel)
}


#cross validation
kfoldCV<-function(traindat,nfold,k=1:15,trainlabel,dismethod=c("mahalanobis","euclidean","manhattan"))
{require(StatMatch)
 random=sample(1:nrow(traindat))
 group=split(random,cut(1:nrow(traindat),nfold))
 if(dismethod=="mahalanobis")
   dmatr=mahalanobis.dist(traindat)
 if(dismethod=="euclidean")
   dmatr=as.matrix(dist(traindat))
 if (dismethod=="manhattan")
   dmatr=as.matrix(dist(traindat,method="manhattan"))
 a=k
 for(i in k){
   err=sapply(group, function(x,dmatr,label){
     pred=KNN(matr=dmatr[x,-x],k=i,trainlabel=label[-x])
     sum(pred!=label[x])/length(pred)
   },dmatr=dmatr,label=trainlabel)
   a[i]=mean(err)
 }
 list(k=which.min(a),minerr=min(a),df=data.frame(ks=k,errs=a))
}
#put 2 group into a subset 
# use 5 folds  cross validation to find the best k for each pair
#you may not want to run this ,very slow
CV=clusterApply(cl,v,function(x,trian,nfolds) 
  sapply(x,function(t,train,nfolds) {
    dat=train[train$label==t[[1]]|train$label==t[[2]],]
kfoldCV(traindat=dat[,-1],nfold=nfolds,trainlabel=dat$label,dismethod="euclidean")
},
train=train,nfolds=nfolds),train=training,nfolds=5
)
dat12=training[training$label==1|training$label==2,]
dat23=training[training$label==2|training$label==3,]
dat13=training[training$label==1|training$label==3,]
dat14=training[training$label==1|training$label==4,]
dat24=training[training$label==2|training$label==4,]
dat34=training[training$label==3|training$label==4,]
cv12=kfoldCV(traindat=dat12[,-1],nfold=5,trainlabel=dat12$label,dismethod="euclidean")
cv23=kfoldCV(traindat=dat23[,-1],nfold=5,trainlabel=dat23$label,dismethod="euclidean")
cv13=kfoldCV(traindat=dat13[,-1],nfold=5,trainlabel=dat13$label,dismethod="euclidean")
cv14=kfoldCV(traindat=dat14[,-1],nfold=5,trainlabel=dat14$label,dismethod="euclidean")
cv24=kfoldCV(traindat=dat24[,-1],nfold=5,trainlabel=dat24$label,dismethod="euclidean")
cv34=kfoldCV(traindat=dat34[,-1],nfold=5,trainlabel=dat34$label,dismethod="euclidean")
a=cbind(cv12$df,cv13$df,cv14$df,cv23$df,cv24$df,cv34$df)
a=a[,c(-3,-5,-7,-9,-11)]


#predict the testing, save the result in a list
#taking very long time
Binaryfit<-function(train, test, trlabel,ngroup=4,
                 seq=c(1,8,1,1,4,3))
{predlist=list()
 pred=list()
 p=1
 for(i in 1:(ngroup-1)){
   for(j in (i+1):ngroup){
     pred[[j]]=KNN(testdat=test,
  traindat=train[train$label==i|train$label==j,][,-1],k=seq[p],trainlabel=train$label,
  dismethod="euclidean");p=p+1}
   predlist[[i]]=pred}
 predlist
}
m=Binaryfit(train=training,test=testing[,-1])

a
#vote
mt=matrix(rep(0,6*4172),ncol=6)
p=1
name1=rep(0,6)
for(i in 1:3)
  for(j in (i+1):4){
    mt[,p]=sapply(m[[i]][[j]],function(x) ifelse(x==i,1,-1))
    name1[p]=paste(as.character(i),as.character(j))
    p=p+1
  }


newmt=as.data.frame(mt)
names(newmt)=name1
head(newmt)
pred=apply(newmt,1,function(x) {m=matrix(rep(0,36),6) 
                                for(i in 1:5) 
                    m[,i]=c(rep(0,i),x[(-0.5*i^2+6.5*i-5):(-0.5*i^2+5.5*i)])
                                m=t(m+t(-m))
                                which.max(apply(m,1,sum))
})
y=KNN(testdat=testing[,-1],traindat=training[,-1],
      trainlabel=training[,1],k=1,dismethod="euclidean")
table(y,testing[,1])
table(pred,testing[,1])