#SOURCE: https://tradewithpython.com/news-sentiment-analysis-using-python

#pip install newspaper3k
#pip install GoogleNews
#pip install nltk
#pip install newspaper
#pip install wordcloud

######################################################################
#GLOBALS
######################################################################

MAX_DAYS_LOOKBACK = 8
DEBUG = 2 #Set to 0 to iterate over all stocks, 1 to use MAX_STOCK_ANALYSIS_NUMBER, 2 to use predefined TICKER_SYMBOLS
TICKER_SYMBOLS = ['AMZN'] #Predefined tickers if DEBUG is 2
MAX_STOCK_ANALYSIS_NUMBER = 2 #If DEBUG > 1, this value is used to set how many stocks will be analyzed before exiting

#Multiplier added to Sentiment. Neutral Sentiment is treated as positive.
SENTIMENT_BIAS_FACTOR_POSITIVE = 1
SENTIMENT_BIAS_FACTOR_NEUTRAL  = 0.25
SENTIMENT_BIAS_FACTOR_NEGATIVE = -1

######################################################################
#IMPORTS
######################################################################

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS
import json

#nltk.download('vader_lexicon') #required for Sentiment Analysis
#nltk.download('punkt')

######################################################################
# Utils
######################################################################

# Parse All tickers from the SEC
def get_ticker_symbols():
    with open('tickers_sec.json', 'r') as tickers_file:
        data_json = json.load(tickers_file)
    ticker_list = []
    for key, value in data_json.items():
        ticker_list.append(value['ticker'])
    return ticker_list

#Display Sentiment of a single stock in a Pie Graph
def pie_chart(name, positive, neutral, negative):
    #Creating PieCart
    labels = ['Positive ['+str(round(positive))+'%]' , 'Neutral ['+str(round(neutral))+'%]','Negative ['+str(round(negative))+'%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'blue','red']
    patches, texts = plt.pie(sizes,colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title("Sentiment Analysis Result for stock= "+name+"" )
    plt.axis('equal')
    plt.show()

# Word cloud visualization
def word_cloud(text):
    stopwords = set(STOPWORDS)
    allWords = ' '.join([nws for nws in text])
    wordCloud = WordCloud(background_color='black',width = 1600, height = 800,stopwords = stopwords,min_font_size = 20,max_font_size=150,colormap='prism').generate(allWords)
    fig, ax = plt.subplots(figsize=(20,10), facecolor='k')
    plt.imshow(wordCloud)
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.show()

######################################################################
# Processing
######################################################################

def process_sentiment(tickerSymbol, startDate, endDate):

    global SENTIMENT_BIAS_FACTOR_POSITIVE
    global SENTIMENT_BIAS_FACTOR_NEUTRAL
    global SENTIMENT_BIAS_FACTOR_NEGATIVE
    
    #---------------------
    #DEBUG PRINT STATEMENT
    #---------------------
    #print('Processing: ' + tickerSymbol)
    
    user_agent = 'Mozilla/5.0 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10
    #As long as the company name is valid, not empty...
    if tickerSymbol != '':
        #print(f'Searching for and analyzing {tickerSymbol}, Please be patient, it might take a while...')

        #Extract News with Google News
        googlenews = GoogleNews(start=startDate,end=endDate)
        googlenews.search(tickerSymbol)
        result = googlenews.result()
        #store the results
        df = pd.DataFrame(result)
        
        #---------------------------
        #View Web scraping results
        #---------------------------
        #print(df)

    #Summarizing
    try:
        list =[] #creating an empty list 
        for i in df.index:
            dict = {} #creating an empty dictionary to append an article in every single iteration
            article = Article(df['link'][i],config=config) #providing the link
            try:
              article.download() #downloading the article 
              article.parse() #parsing the article
              article.nlp() #performing natural language processing (nlp)
            except:
               pass 
            #storing results in our empty dictionary
            dict['Date']=df['date'][i] 
            dict['Media']=df['media'][i]
            dict['Title']=article.title
            dict['Article']=article.text
            dict['Summary']=article.summary
            dict['Key_words']=article.keywords
            list.append(dict)
        check_empty = not any(list)
        # print(check_empty)
        if check_empty == False:
          news_df=pd.DataFrame(list) #creating dataframe
          
          #-----------------
          # Print News Hits
          #-----------------
          #print(news_df)

    except Exception as e:
        #exception handling
        print("exception occurred:" + str(e))
        print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.' )

        #Sentiment Analysis
        #Sentiment Analysis
    def percentage(part,whole):
        return 100 * float(part)/float(whole)

    #Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    #Creating empty lists
    news_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    #Iterating over the tweets in the dataframe
    for news in news_df['Summary']:
        news_list.append(news)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(news);
        neg = analyzer['neg']
        neu = analyzer['neu']
        pos = analyzer['pos']
        comp = analyzer['compound']

        if neg > pos:
            negative_list.append(news) #appending the news that satisfies this condition
            negative += 1 #increasing the count by 1
        elif pos > neg:
            positive_list.append(news) #appending the news that satisfies this condition
            positive += 1 #increasing the count by 1
        elif pos == neg:
            neutral_list.append(news) #appending the news that satisfies this condition
            neutral += 1 #increasing the count by 1 

    positive = percentage(positive, len(news_df)) #percentage is the function defined above
    negative = percentage(negative, len(news_df))
    neutral = percentage(neutral, len(news_df))

    #Converting lists to pandas dataframe
    news_list = pd.DataFrame(news_list);
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    
    #using len(length) function for counting

    #------------------
    # Sentiment Result
    #------------------

    #Uncomment for Sentiment analysis values
    print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
    print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
    print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')

    #Uncomment for a visual of the sentiment 
    #pie_chart(tickerSymbol, positive, neutral, negative)

    #Uncomment for wordcloud info, but not very useful
    #print('Wordcloud for ' + tickerSymbol)
    #word_cloud(news_df['Summary'].values)







    #----------------------------
    # AI Tuning Here
    #----------------------------
    
    totalHits = len(positive_list) + len(neutral_list) + len(negative_list)
    if totalHits == 0:
        totalHits = 1 #Divide by 0 prevention
    else:
        positiveSentiment = len(positive_list) / totalHits * SENTIMENT_BIAS_FACTOR_POSITIVE
        neutralSentiment = len(neutral_list) / totalHits * SENTIMENT_BIAS_FACTOR_NEUTRAL
        negativeSentiment = len (negative_list) / totalHits * SENTIMENT_BIAS_FACTOR_NEGATIVE

    retVal = positiveSentiment + neutralSentiment + negativeSentiment
    print("Positive ......:",  retVal, end='\n')
    return retVal



def GenerateCSVFile(ticker_symbol,CSVFileName,NumOfDays):

    global DEBUG
    global MAX_STOCK_ANALYSIS_NUMBER
    global MAX_DAYS_LOOKBACK
    global TICKER_SYMBOLS
    import os
   
    
    #now = dt.date.today()
    #endDate = now.strftime('%m-%d-%Y') #Set End date to today
    
    StockSentimentMap = {}
    sentiment_data = {}
    stockCounter = 0
    end_date = dt.date.today()
     # Iterate over the last week
    #for i in range(1, 3):
    for i in range(1, NumOfDays):
        start_date = end_date - dt.timedelta(days=i)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        
        print(f"Sentiment analysis for the period: {start_date_str} to {end_date_str}")
        
        StockSentimentMap = {}
        #for ticker_symbol in TICKER_SYMBOLS:
        sentiment = process_sentiment(ticker_symbol, start_date_str, end_date_str)
            #Normalize sentiment value to either '1' or '0'
        if(sentiment >=0.7):
                sentiment= 1
        else:
                sentiment= 0

        StockSentimentMap[ticker_symbol] = sentiment
        print()  # Add a newline for clarity
        
        
        df = pd.DataFrame(StockSentimentMap.items(), columns=['Ticker', 'Sentiment'])
        df['Date'] = start_date_str
        df = df[['Date', 'Ticker', 'Sentiment']]
        
        # Check if the CSV file exists
        #csv_file = 'sentiment_data.csv'
        csv_file = CSVFileName
        if os.path.exists(csv_file):
            # Append to existing CSV file
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            # Create a new CSV file with headers
            df.to_csv(csv_file, index=False)






