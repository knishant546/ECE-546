#!/usr/bin/env python
# coding: utf-8

######################################################################
#GLOBALS
######################################################################
USING_JUPYTER = False #Enable to use get_ipython() functions
SENTIMENT_ANALYSIS = False #Enable to include news analysis
NUMBER_OF_TICKERS_BEFORE_EXIT = 10 #If -1, all stocks will be traversed. 10 is reasonable
DEBUG = False #Will enable all command prompts, plots, and figures

if USING_JUPYTER:
    get_ipython().system('pip install yfinance')
    get_ipython().system('pip install ipynb')

    if SENTIMENT_ANALYSIS:
        get_ipython().system('pip install newspaper3k')
        get_ipython().system('pip install GoogleNews')
        get_ipython().system('pip install nltk')
        get_ipython().system('pip install newspaper')
        get_ipython().system('pip install wordcloud')
        pass
else:
    #Paul needs this to avoid Py4JJavaError (hadoop env var issue?)
    import findspark
    findspark.init()  

######################################################################
#IMPORTS
######################################################################

from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as sqlFn
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import datetime as dt
import json
from pathlib import Path
from operator import itemgetter
import time

if SENTIMENT_ANALYSIS:
    from SentimentLib import process_sentiment
    
######################################################################
#FUNCTIONS
######################################################################

def main():  
    global NUMBER_OF_TICKERS_BEFORE_EXIT
    global DEBUG
    stock_dict = dict()
    print("Running Stock Prediction Model")
    tickerSymbolList = get_ticker_symbols() #Note, there are 10423 tickers, however it looks like the SEC ranks them. So maybe only first 100 are really useful
    
    print("Opening CMD Prompt to create Spark Session")
    spark = SparkSession.builder.appName("StockPredictionModel").getOrCreate()
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    currentDate = datetime.now()
    
    #Perform Predictions for the past 30 days
    #for j in range(0, 31):
    for j in range(0,31):
        predictionDate = currentDate - timedelta(days=j)
        predictionDateStr = predictionDate.strftime('%Y-%m-%d')
        print(predictionDateStr)

        #Find which dates are weekdays
        weekno = predictionDate.weekday()
        if (weekno >= 5):
            print("Skipping Weekend")
            continue #Don't perform Analysis on weekend day since stock doesn't change

        #Process each stock based on the ticker_sec.json list
        for i in range(0, len(tickerSymbolList)):
            
            #Due to there being over 10000 stocks, an option is added to limit stock analysis to the first 1-10 stocks in the sec list (or more)
            if (NUMBER_OF_TICKERS_BEFORE_EXIT > 0) and (i >= NUMBER_OF_TICKERS_BEFORE_EXIT):
                break #Premature exit
            else:
                print(str(i+1) + ": " + tickerSymbolList[i])

                #return of process_stock is a tuple: (recentActualClose, predictionValue, rmse, actualCloseVsOpen, recentActualOpen)
                retVal = process_stock(spark, tickerSymbolList[i], predictionDate)                
                stock_dict[tickerSymbolList[i]] = retVal #Correlate tuple with stock by using a dictionary
        #End iterate over tickerSymbolList

        #Run Sentiment analysis gains on each predicted vs actual close stock prices
        evaluatedStock_dict = process_best_worst_stock(stock_dict)
        
        bestPredictedStockToBuy = max(evaluatedStock_dict, key=evaluatedStock_dict.get) #highest predicted delta is the best to buy
        bestPredictedStockToSell = min(evaluatedStock_dict, key=evaluatedStock_dict.get) #lowest predicted (or negative) delta is the best to sell
        print("BEST PREDICTED STOCK TO BUY: " + str(bestPredictedStockToBuy) + " | Value: " + str(evaluatedStock_dict[bestPredictedStockToBuy]))
        print("BEST PREDICTED STOCK TO SELL: " + str(bestPredictedStockToSell) + " | Value: " + str(evaluatedStock_dict[bestPredictedStockToSell]))

        bestActualStockToBuy = max(stock_dict[key][3] for key in stock_dict)
        bestActualStockToSell = min(stock_dict[key][3] for key in stock_dict)

        print("BEST ACTUAL STOCK TO BUY: " + str(bestActualStockToBuy))
        print("BEST ACTUAL STOCK TO SELL: " + str(bestActualStockToSell))

        #Open File to store actual and model prediction data
        file = open("./predictions/prediction_" + predictionDateStr + ".csv", "w")
        file.write("Stock,PredictedClose,ActualOpen,ActualClose,RMSE,SentimentValue,PredictedStockToBuy,PredictedStockToSell,ActualStockToBuy,ActualStockToSell,ActualDailyChange\n")
    
        #both evaluatedStock_dict and stock_dict are the same size.
        #iterate over all evaluated stocks so data can be output to a file
        for stock in evaluatedStock_dict:
            actualClose = stock_dict[stock][0]
            predictedValue = stock_dict[stock][1]
            rmse = stock_dict[stock][2]
            sentimentValue = evaluatedStock_dict[stock]
            actualDailyChange = stock_dict[stock][3]
            actualOpen = stock_dict[stock][4]

            predictedStockToBuy = 0
            predictedStockToSell = 0
            if(stock == bestPredictedStockToBuy):
                predictedStockToBuy = 1
            if(stock == bestPredictedStockToSell):
                predictedStockToSell = 1

            actualStockToBuy = 0
            actualStockToSell = 0
            if(actualDailyChange == bestActualStockToBuy):
                actualStockToBuy = 1
            if(actualDailyChange == bestActualStockToSell):
                actualStockToSell = 1
                
            file.write(stock + "," + str(predictedValue) + "," + str(actualOpen) + "," + str(actualClose) + "," + str(rmse) + "," + str(sentimentValue) + "," + str(predictedStockToBuy) + "," + str(predictedStockToSell) + "," +  str(actualStockToBuy) + "," + str(actualStockToSell) + "," + str(actualDailyChange) + "\n")
            
        file.close()
        #end iterate number of days loop
        
    # Terminate the Spark session
    spark.stop()

# Parse All tickers from the SEC
def get_ticker_symbols():
    with open('tickers_sec.json', 'r') as tickers_file:
        data_json = json.load(tickers_file)
    ticker_list = []
    for key, value in data_json.items():
        ticker_list.append(value['ticker'])
    return ticker_list

#Process Stock based on ticker
def process_stock(spark, stock_ticker, predictionDate):
    stock = yf.Ticker(stock_ticker)
    # GET TODAYS DATE AND CONVERT IT TO A STRING WITH YYYY-MM-DD FORMAT (YFINANCE EXPECTS THAT FORMAT)
    end_date = predictionDate
    end_date_str = predictionDate.strftime('%Y-%m-%d')
    
    stock_hist = stock.history(start='2010-01-16',end=end_date)
    
    stockHistoryFile = 'stock_histories/' + stock_ticker + '.csv'
    stock_hist.to_csv(stockHistoryFile)

    df = spark.read.csv(stockHistoryFile, header=True, inferSchema=True)
    df1 = df

    if DEBUG:
        #Display stock data
        df1.show(8)

    # drop any row having any Null 
    df = df.dropna(how="any")

    # openCloseChange
    df = df.withColumn("openCloseChange", (df.Close - df.Open) / df.Open)

    # maxDayChange
    df = df.withColumn("maxDayChange", df.High - df.Low)

    # dividend provided
    df = df.withColumn("dividend", sqlFn.when(df["Dividends"] > 0, 1).otherwise(0))

    # Stock split
    df = df.withColumn("stockSplit", sqlFn.when(df["Stock Splits"] != 1, 1).otherwise(0))

    # order by date 
    w = Window.partitionBy().orderBy("date")

    # Lagged column for the 'close' price (i.e., previous day's close)
    df = df.withColumn("lagClose", sqlFn.lag(df.Close).over(w))

    #  DailyChange - change in closing price from the previous day
    df = df.withColumn("DailyChange", df.Close - df.lagClose)

    # moving average for the closing prices
    df = df.withColumn("movingAvgClose", sqlFn.avg(df.Close).over(w.rowsBetween(-6, 0)))

    # drop any row having any Null 
    df = df.dropna(how="any")

    consolidatedFeature = ["Open", "High", "Low", "Close", "Volume", "openCloseChange", 
                   "maxDayChange", "DailyChange", "movingAvgClose", 
                   "dividend", "stockSplit"]

    #store features in the vector column
    assembler = VectorAssembler(inputCols=consolidatedFeature, outputCol="features")
    df_assembled = assembler.transform(df)

    # Split the data into a training set - 80% , 20% test set. 

    trainingDataCount = int(df_assembled.count() * 0.8)
    trainingData = df_assembled.orderBy("date").limit(trainingDataCount)
    testData = df_assembled.subtract(trainingData)

    # GBT Model Training
    gbt = GBTRegressor(labelCol="Close", featuresCol="features", maxIter=10, maxBins =64, maxDepth =5,stepSize =0.25)

    model = gbt.fit(trainingData)

    predictions = model.transform(testData)


    # Model Evaluation
    # Compute the RMSE (Root Mean Squared Error) for the predictions
    evaluator_rmse = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")


    rmse = evaluator_rmse.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data =", rmse)


    # Mean Absolute Error (MAE) and R-squared (R2)
    for metric in ["mae", "r2"]:
        evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName=metric)
        value = evaluator.evaluate(predictions)
        print(f"{metric.upper()}: {value}")

    preds = predictions.select("Date", "Open", "Close", "prediction").toPandas()

    if DEBUG:
        plt.figure(figsize=(12, 6))
        plt.plot(preds["Close"], label='Actual', color='blue')
        plt.plot(preds["prediction"], label='Predicted', color='red', alpha=0.6)
        plt.title('Actual Close vs Prediction Close')
        plt.show(block=False)

    # Convert "Date" column to datetime type, with utc=True
    preds['Date'] = pd.to_datetime(preds['Date'], utc=True)

    # Filter data for the Last 2 months
    six_months_ago = pd.Timestamp.now(tz='UTC') - timedelta(days=30*2)
    preds_last_6_months = preds[preds['Date'] >= six_months_ago]

    if DEBUG:
        # Creating the plot
        plt.figure(figsize=(12, 6))  # Setting the figure size

        # Plotting actual values as blue bars
        plt.bar(preds_last_6_months.index, preds_last_6_months["Close"], color='blue', width=0.6, label='Actual')

        # Shifting the position of predicted values slightly to the right for better visualization
        plt.bar(preds_last_6_months.index + 0.4, preds_last_6_months["prediction"], color='red', width=0.6, label='Predicted')

        plt.xlabel('Index')  # Labeling x-axis
        plt.ylabel('Close Price')  # Labeling y-axis
        plt.title('Actual vs Predicted Adjusted Closing Prices for Last 6 Months')  # Setting the title of the plot
        plt.legend()  # Showing the legend
        plt.xticks(preds_last_6_months.index + 0.2, preds_last_6_months['Date'].dt.strftime('%Y-%m-%d'))  # Setting x-axis ticks with dates
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjusting layout for better visualization

        print("Displaying Actual vs Predicted Adjusted Closing prices for Last 6 Months")
        plt.show(block=False)  # Displaying the plot

    # Prepare historical data for the past 1 month
    end_date_past = end_date_str
    start_date_past = (end_date - timedelta(days=30)).strftime('%Y-%m-%d')
    past_data = df.filter((sqlFn.col("Date") >= start_date_past) & (sqlFn.col("Date") <= end_date_past))

    # Apply the same feature engineering steps to past_data
    # Assuming you've defined the feature_columns as before
    consolidatedFeature = ["Open", "High", "Low", "Close", "Volume", "openCloseChange", 
                   "maxDayChange", "DailyChange", "movingAvgClose", 
                   "dividend", "stockSplit"]

    # Assemble features
    assembler = VectorAssembler(inputCols=consolidatedFeature, outputCol="features")
    past_data_assembled = assembler.transform(past_data)

    # Apply the trained model to make predictions for the past 1 month
    past_predictions = model.transform(past_data_assembled)

    # Plot the historical and predicted data for the past 1 month
    past_data_pd = past_data.select("Date", "Close").toPandas()
    past_pred_pd = past_predictions.select("Date", "prediction").toPandas()

    last_date = datetime.strptime(end_date_past, '%Y-%m-%d')

    # Prepare future data for the next 7 days using the last available data point
    end_date_future = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    future_dates = [last_date + timedelta(days=i) for i in range(1, 7)]  # Include next 7 days
    future_df = spark.createDataFrame([(d,) for d in future_dates], ["Date"])
    last_data_point = df.orderBy("Date", ascending=False).limit(1)  # Get the last available data point
    future_df = future_df.crossJoin(last_data_point.drop("Date"))

    # Apply the same feature engineering steps to future_df
    future_df = future_df.withColumn("lagClose", sqlFn.lag(future_df.Close).over(Window.orderBy("Date")))
    future_df = future_df.withColumn("dayChange", (future_df.Close - future_df.Open) / future_df.Open)
    future_df = future_df.withColumn("maxDayChange", future_df.High - future_df.Low)
    future_df = future_df.withColumn("DailyChange", future_df.Close - future_df.lagClose)
    future_df = future_df.withColumn("movingAvgClose", sqlFn.avg(future_df.Close).over(Window.rowsBetween(-6, 0)))
    future_df = future_df.withColumn("dividend", sqlFn.when(future_df["Dividends"] > 0, 1).otherwise(0))
    future_df = future_df.withColumn("stockSplit", sqlFn.when(future_df["Stock Splits"] != 1, 1).otherwise(0))
    future_df = future_df.dropna()
    future_df_assembled = assembler.transform(future_df)

    # Apply the trained model to make predictions for the next 7 days
    print("Performing Future Predictions Model Transform")
    future_predictions = model.transform(future_df_assembled)

    # Plot the predicted data for the next 7 days
    future_pred_pd = future_predictions.select("Date", "prediction").toPandas()

    # Convert date column to pandas datetime object
    future_pred_pd["Date"] = pd.to_datetime(future_pred_pd["Date"])

    # Plot the predicted data for the next 7 days
    if DEBUG:
        plt.figure(figsize=(12, 6))
        plt.plot(future_pred_pd["Date"], future_pred_pd["prediction"], label='Predicted Close (Next 30 Days)', color='green', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Predicted Close Prices (Next 30 Days)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        print("Displaying Predicted Close Prices (Next 30 Days)")
        plt.show(block=False)

    #Get the latest real close value
    close_last_6_months = preds_last_6_months["Close"]
    recentActualCloseList = close_last_6_months.tolist()
    recentActualClose = recentActualCloseList[len(recentActualCloseList)-1]

    #Get the latest  real open value
    open_last_6_months = preds_last_6_months["Open"]
    recentActualOpenList = open_last_6_months.tolist()
    recentActualOpen = recentActualOpenList[len(recentActualOpenList)-1]

    #Calculate daily change on lastest day
    actualCloseVsOpen = recentActualClose - recentActualOpen

    #Return Prediction Value
    predictionValue = future_pred_pd["prediction"][len(future_pred_pd["prediction"])-1]
    
    return((recentActualClose, predictionValue, rmse, actualCloseVsOpen, recentActualOpen))

#Return a dictionary with stock as a key of the values (Predicted Value - LastDayClose) * Sentiment
#The highest value means the best to buy. The Lowest value means the best to sell.
def process_best_worst_stock(stock_dict):
    evaluatedStock_dict = dict()
    print("Performing Best/Worst Stock Analysis")
    if(SENTIMENT_ANALYSIS):
        print("Using Sentiment Analysis")
        
    for stock in stock_dict:
        lastClosed = stock_dict[stock][0]
        predicted = stock_dict[stock][1]

        #Get delta between predicted value and the last day the stock closed
        stockDelta = predicted - lastClosed

        #Perform Sentiment Analysis as a gain factor
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=7) #just look at the past week
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
                
        
        if SENTIMENT_ANALYSIS:
            sentiment = process_sentiment(stock, start_date_str, end_date_str)
            #If predicted stock is a gain, Normalize sentiment value between .5=negative, 1=neutral, 1.5=positive
            #This will cause the good delta value to maximize if positive or minimize if negative
            if stockDelta >= 0:
                if(sentiment >=0.7):
                    sentiment= 1.5
                elif sentiment >=0.3:
                    sentiment= 1
                else:
                    sentiment = .5

            #If predicted stock is a loss, do the inverse: 1.5=negative, 1=neutral, .5=positive
            #This will cause negative to maximize the bad delta value and a positive sentiment will minimize the bad delta
            else:
                if(sentiment >=0.7):
                    sentiment= .5
                elif sentiment >=0.3:
                    sentiment= 1
                else:
                    sentiment = 1.5

            #Use sentiment as a gain to the stock delta
            stockDelta = stockDelta * sentiment

        evaluatedStock_dict[stock] = stockDelta

    return evaluatedStock_dict

######################################################################
#MAIN ENTRY
######################################################################

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Elapsed Time: " + str(end - start))
