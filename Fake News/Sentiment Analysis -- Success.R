#### Read in Libraries and raw data ####
library(tidyverse)
library(tidytext)
library(textdata)
library(dplyr)
library(DataExplorer)
library(caret)

# Read in raw data and combine to form all_articles
train <- read_csv("train.csv")
test <- read_csv("test.csv")

all_articles <- bind_rows(train = train, test = test, .id = "Set")
all_articles$label <- as.factor(all_articles$label)

# Simplify the data to only include what we care about
article_df <- all_articles %>% select(id, text)

#### Create Sentiment Values based off words in articles ####
# (Only works for English words) #
article_sentiment <- article_df %>% 
  unnest_tokens(token = "words", input = text, output = word) %>% 
  inner_join(get_sentiments("afinn"))

article_sentiment2 <- article_df %>%
  unnest_tokens(token = "words", input = text, output = word) %>%
  inner_join(get_sentiments("bing")) %>%
  mutate(sentiment = ifelse(sentiment == "negative", -1, 1))

article_sentiment3 <- article_df %>%
  unnest_tokens(token = "words", input = text, output = word) %>%
  inner_join(get_sentiments("nrc")) %>%
  mutate(sentiment = as.factor(sentiment))
levels(article_sentiment3$sentiment) <- c(-1, 0, -1, -1, 1, -1, 1, -1, 0, 1) # give factors numeric score (subjective)
article_sentiment3$sentiment2 <- varhandle::unfactor(article_sentiment3$sentiment)  # make factors numbers

article_sentiment4 <- article_df %>%
  unnest_tokens(token = "words", input = text, output = word) %>%
  inner_join(get_sentiments("loughran")) %>% 
  mutate(sentiment = as.factor(sentiment))
levels(article_sentiment4$sentiment) <- c(0, 0, -1, 1, 0, 0)
article_sentiment4$sentiment2 <- varhandle::unfactor(article_sentiment4$sentiment)

#### Join all sentiment scores together ####

all_articles <- all_articles %>%
  left_join(article_sentiment %>%
              group_by(id) %>%
              summarise(value = sum(value))) %>%
  left_join(article_sentiment2 %>% 
              group_by(id) %>% 
              summarise(sentiment = sum(sentiment))) %>% 
  left_join(article_sentiment3 %>%
              group_by(id) %>%
              summarise(sentiment2 = sum(sentiment2))) %>%
  left_join(article_sentiment4 %>%
              group_by(id) %>%
              summarise(sentiment3 = sum(sentiment2))) %>%
  replace_na(list(value = 0, sentiment = 0, sentiment2 = 0, sentiment3 = 0)) %>% 
  mutate(label = ifelse(label == 0, "Fake", "True"))

# Subset data so we only have what we care about
sentiment_scores <- all_articles %>% 
  select(-c(title, author, text))

#### Box Plots ####

library(ggplot2)

box1 <- ggplot(sentiment_scores %>% filter(Set == "train"), aes(label, value, color = label)) +
  geom_boxplot() +
  scale_y_continuous(limits = c(-100, 100)) + # Zoomed in to see more difference
  theme(legend.position = "none") +
  theme_classic()

box2 <- ggplot(sentiment_scores %>% filter(Set == "train"), aes(label, sentiment, color = label)) +
  geom_boxplot() +
  scale_y_continuous(limits = c(-100, 100)) + # Zoomed in to see more difference
  theme(legend.position = "none") +
  theme_classic()

box3 <- ggplot(sentiment_scores %>% filter(Set == "train"), aes(label, sentiment2, color = label)) +
  geom_boxplot() +
  scale_y_continuous(limits = c(-100, 100)) + # Zoomed in to see more difference
  theme(legend.position = "none") +
  theme_classic()

box4 <- ggplot(sentiment_scores %>% filter(Set == "train"), aes(label, sentiment3, color = label)) +
  geom_boxplot() +
  scale_y_continuous(limits = c(-100, 100)) + # Zoomed in to see more difference
  theme(legend.position = "none") +
  theme_classic()

ggpubr::ggarrange(box1, box2, box3, box4, common.legend = TRUE, legend = "right")

#### Modeling ####

# Split the data back to train and test
train.sent <- sentiment_scores %>% filter(!is.na(label))
test.sent <- sentiment_scores %>% filter(is.na(label))

# Some additional Libraries
library(plyr)
library(mboost)

# Boosted Linear Model
bglm <- train(form = label ~ .,
              data = train.sent %>% select(-c(Set, id)),
              method = "glmboost",
              trControl = trainControl(method = "repeatedcv", classProbs = TRUE,
                                       number = 10, repeats = 3),
              metric = "Accuracy",
              tuneGrid = expand.grid(mstop = 1, prune = TRUE))

# Create and write our predictions
sent.pred <- data.frame(id = test.sent$id, label = predict(bglm, newdata = test.sent))
sent.pred <- sent.pred %>%  mutate(label = ifelse(label == "Fake", 0, 1))
write.csv(sent.pred, "sentiment_predictions.csv", row.names = FALSE)
