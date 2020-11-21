# Load Libraries and the data
library(tidyverse)
library(caret)
library(nnet)

train <- read_csv("train.csv")
test <- read_csv("test.csv")

full <- bind_rows(train = train, test = test, .id = "Set")
#79146 x 129

# Subset explanatory variables
just_a_look <- c("Ins_Age", "Ht", "Wt", "BMI", "Set", "Response")
interpretable <- full %>% select(all_of(just_a_look))

# EDA
hist(full$Response)
GGally::ggpairs(interpretable %>% filter(Set == "train") %>% select(-Set))
cor(interpretable %>% filter(Set == "train") %>% select(-Set))
summary(interpretable)

# Multi-Nomial model
interpretable <- interpretable %>% mutate(Response = as.factor(Response))

multi_nom <- train(form = Response ~ .,
                   data = interpretable %>% filter(Set == "train") %>% select(-Set),
                   method = "multinom",
                   trControl = trainControl(method = "repeatedcv", number = 10),
                   tuneGrid = expand.grid(decay = .0001)) # From Best Tune

multi_nom$results

# Export Results
submission <- data.frame(Id = test %>% pull(Id),
                         Response = predict(multi_nom, newdata = interpretable %>% 
                                              filter(Set == "test") %>% 
                                              select(-Set)))
write.csv(submission, "submission.csv", row.names = FALSE)
