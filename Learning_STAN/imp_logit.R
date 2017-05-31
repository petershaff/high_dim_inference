library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

dat <- read.csv('train.csv')
train <- as.data.frame(cbind(dat$Survived, dat$Pclass, dat$Sex, dat$Age))
names(train) <- c(
