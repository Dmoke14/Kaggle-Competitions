test <- read.csv("test (1).csv")
train <- read.csv("train.csv")
library(tidyverse)
library(DataExplorer)

#Explore data
head(train)
str(train$date)

#Merge data 
store <- bind_rows(train=train, test=test, .id="Set")
store <- store[,1:5]

store$date <- as.Date(store$date, "%Y-%m-%d")

plot_missing(store)

#Create month and year variabel
store <- store %>%
  mutate(month = format(date, "%m"), year = format(date, "%Y"))

store$wday <- weekdays(store$date)

season <- function(x){
  if ((x > '0' & x < '3') | x == '12') {
    szn <- "Winter"
  }
  else if ((x > '2' & x < '6')) {
    szn <- "Spring"
  }
  else if ((x > '5' & x < '9')) {
    szn <- "Summer"
  }
  else {
    szn <- "Fall"
  }
  return(szn)
}

for (i in 1:length(store)) {
  store$season[i] <- season(store$month[i])
}
