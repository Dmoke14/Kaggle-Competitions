##### Tried to fit a CORPUS. Ran into some issues. #####

library(tidyverse)
library(dplyr)
library(DataExplorer)
library(caret)

train <- read_csv("train.csv")
test <- read_csv("test.csv")

library(tm)
library(SnowballC)
library(SentimentAnalysis)

#Create a corpus from a random set of articles
random_articles <- sample(c(1:20800), 2800)
train.corpus <- SimpleCorpus(VectorSource(train$text[random_articles]))
test.corpus <- SimpleCorpus(VectorSource(test$text))
# 1. Stripping any extra white space:
train.corpus <- tm_map(train.corpus, stripWhitespace)
test.corpus <- tm_map(test.corpus, stripWhitespace)
# 2. Transforming everything to lowercase
# train.corpus <- tm_map(train.corpus, content_transformer(tolower)) # Other languages -- Can't work
# 3. Removing numbers 
train.corpus <- tm_map(train.corpus, removeNumbers)
test.corpus <- tm_map(test.corpus, removeNumbers)
# 4. Removing punctuation
train.corpus <- tm_map(train.corpus, removePunctuation)
test.corpus <- tm_map(test.corpus, removePunctuation)
# 5. Removing stop words
train.corpus <- tm_map(train.corpus, removeWords, stopwords("english"))
test.corpus <- tm_map(test.corpus, removeWords, stopwords("english"))

train.corpus <- tm_map(train.corpus, stemDocument)
test.corpus <- tm_map(test.corpus, stemDocument)

train.DTM <- DocumentTermMatrix(train.corpus)
test.DTM <- DocumentTermMatrix(test.corpus)
# inspect(train.DTM)
# inspect(test.DTM)

train.sent <- as.data.frame(analyzeSentiment(train.DTM, language = "english")[, 1:4])
test.sent1 <- as.data.frame(analyzeSentiment(test.DTM, language = "english")[, 1:4])

train.sent <- train.sent %>% 
  mutate(id = train$id[random_articles], label = train$label[random_articles]) %>% 
  filter(!is.na(SentimentGI))

train.sent <- train.sent %>% 
  mutate(label = ifelse(label == 0, "No", "Yes"))

test.sent <- test.sent %>% 
  mutate(id = test$id)

# train.lm <- lm(label ~ SentimentGI + NegativityGI + PositivityGI + WordCount, data = train.sent)

library(plyr)
library(mboost)

bglm <- train(form = label~.,
              data = (train.sent %>% select(-id)),
              method = "glmboost",
              trControl = trainControl(method = "repeatedcv", classProbs = TRUE,
                                       number = 10),
              metric = "Accuracy",
              tuneGrid = expand.grid(mstop = c(1:100), prune = TRUE))

bglm$bestTune
bglm$results[bglm$bestTune[,1],]

# sent.pred <- data.frame(Id = test.sent$id, Predicted = predict(bglm, newdata = test.sent))
# sent.pred <- sent.pred %>%  mutate(label = ifelse(label == "No", 0, 1))
