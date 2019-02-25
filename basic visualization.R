data <- as.data.frame(read.table("C:/Users/robot/Downloads/project2/data/data.txt"))
genre <- as.data.frame(read.table("C:/Users/robot/Downloads/project2/data/movie.txt"))

userID=data[,1]
movieID=data[,2]
rating=data[,3]

#Hist: all rating
hist(rating,breaks=c(0.5,1.5,2.5,3.5,4.5,5.5))

#Hist: 10 most popular movies
uniq_movies=unique(movieID)
frequencies <- matrix(0,nrow=length(uniq_movies),ncol=1)
for (i in 1:length(uniq_movies)){
  temp <- data[data$V2==i,]
  frequencies[i]=length(temp[,1])
}

freq=as.data.frame(cbind(1:length(uniq_movies),frequencies))
ordered_freq=freq[order(freq$V2,decreasing=TRUE),]
top10=ordered_freq[1:10,1]
filtered1=data.frame()
for (i in 1:10){
  ind=top10[i]
  temp=data[data$V2==ind,]
  filtered1=rbind(filtered1,temp)
}
hist(filtered1$V3,breaks=c(0.5,1.5,2.5,3.5,4.5,5.5))

#Hist: 10 highest-rated movies
score <- matrix(0,nrow=length(uniq_movies),ncol=1)
for (i in 1:length(uniq_movies)){
  temp <- data[data$V2==i,]
  score[i]=mean(temp$V3)
}
scores=as.data.frame(cbind(1:length(uniq_movies),score))
ordered_scores=scores[order(scores$V2,decreasing=TRUE),]
top10_scores=ordered_scores[1:10,1]
filtered2=data.frame()
for (i in 1:10){
  ind=top10_scores[i]
  temp=data[data$V2==ind,]
  filtered2=rbind(filtered2,temp)
}
hist(filtered2$V3,breaks=c(0.5,1.5,2.5,3.5,4.5,5.5))

#Hist: horror movies
horror=genre[genre$V13==1,1]
horror_rating=data.frame()
for (i in 1:length(horror)){
  ind=horror[i]
  temp=data[data$V2==ind,]
  horror_rating=rbind(horror_rating,temp)
}
hist(horror_rating$V3,breaks=c(0.5,1.5,2.5,3.5,4.5,5.5))

#Hist: musical movies
musical=genre[genre$V14==1,1]
musical_rating=data.frame()
for (i in 1:length(musical)){
  ind=musical[i]
  temp=data[data$V2==ind,]
  musical_rating=rbind(musical_rating,temp)
}
hist(musical_rating$V3,breaks=c(0.5,1.5,2.5,3.5,4.5,5.5))

#Hist: comedy movies
comedy=genre[genre$V7==1,1]
comedy_rating=data.frame()
for (i in 1:length(comedy)){
  ind=comedy[i]
  temp=data[data$V2==ind,]
  comedy_rating=rbind(comedy_rating,temp)
}
hist(comedy_rating$V3,breaks=c(0.5,1.5,2.5,3.5,4.5,5.5))