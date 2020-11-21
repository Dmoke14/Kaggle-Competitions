# Load packages
library(tidyverse)
# library(DataExplorer)
library(MASS)
library(tseries)
library(forecast)

# Load data
test <- read_csv("test.csv")
train <- read_csv("train.csv")

# Merge data 
store <- bind_rows(train=train, test=test, .id="Set")

# Start ARIMA code https://medium.com/analytics-vidhya/sarima-forecasting-seasonal-data-with-python-and-r-2e7472dfad83
attach(store)

# Create empty vector for updating in the loop
final_fore_vals <- c()

# Create a for loop to generate predictions for each store (assuming store independence)
for (store_num in unique(store$store)) {
  for (item_num in unique(store$item)) {
  
    # Subset the sales
    sales <- store %>%
      filter(Set == "train", store == store_num, item == item_num) %>%
      dplyr::select(sales)
    
    # ACF, PACF and Dickey-Fuller Test
    # acf(sales)
    # pacf(sales)
    # adf.test(sales$sales)
    
    ### ARIMA ###
    
    # Build a Time Series and an ARIMA model
    salesarima <- ts(sales, start = c(2013, 1), frequency = 7)
    fitsales <- auto.arima(salesarima, stepwise = TRUE, test = "kpss", ic = "bic", 
                           max.p = 3, max.P = 2, start.p = 1, start.P = 0)
    
    fitsales
    confint(fitsales)
    
    # plot(salesarima, type = "l")
    
    # Forecast the next 90 test values for each store/item
    fore_val <- forecast(fitsales, 90)
    
    # plot(fore_val)
    
    # Extract the mean values 
    final_fore_vals <- c(final_fore_vals, as.numeric(fore_val$mean))
  }
}

# Combine the predicted value and the id from the test set
fore_test <- data.frame(id = test %>% pull(id), sales = final_fore_vals)
write.csv(fore_test, "submission.csv", row.names = FALSE)