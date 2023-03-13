import pandas as pd
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os, ssl
import requests
import sinopy   
import jieba

# Define the Chinese characters to look for: Tsunami
characters = "海嘯"

# Scrape tsunami news articles
tsunami_articles = ['https://news.ltn.com.tw/news/world/breakingnews/4235092', 'http://news.now.com/home/international/player?newsId=510269', 'https://tw.sports.yahoo.com/news/19-2%E6%B5%B7%E5%98%AF%E6%94%BB%E5%8B%A2-%E5%85%89%E5%BE%A9%E9%AB%98%E4%B8%AD%E6%99%89%E5%86%A0%E8%BB%8D%E8%B3%BD-055111431.html']

# Search for Mandarin articles with the specific characters in the title

for item in tsunami_articles:
    article = Article(item, language="zh")
    article.download()
    article.parse()
    if characters in item:
        tsunami_articles.append(article.text) 

# Scrape non-tsunami news articles
non_tsunami_articles =['https://www.voacantonese.com/a/taiwan-to-allow-more-china-flights-in-show-of-goodwill-030923/6997400.html', 'https://www.thenewslens.com/article/182042']

# Search for Mandarin articles without the specific characters in the title
for item in non_tsunami_articles:
    article = Article(item, language="zh")
    article.download()
    article.parse()
    if characters not in item:  
        non_tsunami_articles.append(article.text)
    
#IDIOMS
    # if the idiom + idiom value are present in the article, search for specific words in the values to double check.
    # add a new classifier to parse these articles in a new bucket. 
# file = open('idioms.txt','r')
# content = file.read().split(',')
# for line in content.readlines():
#     if "海" in line: 
#         non_tsunami_articles.append(article.text)

# Create a Pandas DataFrame
df = pd.DataFrame({'text': tsunami_articles + non_tsunami_articles, 'tsunami': [1]*len(tsunami_articles) + [0]*len(non_tsunami_articles)})

# Preprocess the data
vectorizer = CountVectorizer(stop_words='chinese')
X = vectorizer.fit_transform(df['text'])

# Train the Naive Bayes classifier
y = df['tsunami']
clf = MultinomialNB()
clf.fit(X, y)

# Test the classifier
test_articles = ['https://www.bbc.com/zhongwen/trad/world-64609096', 'https://www.worldjournal.com/wj/story/121186/7024853', 'https://www.worldjournal.com/wj/story/121468/7024421?from=wj_hot_story', 'http://www.cankaoxiaoxi.com/world/20230312/2506753.shtml', 'https://zh.wikipedia.org/zh-hans/%E6%B5%B7%E5%95%B8']

X_test = vectorizer.transform(test_articles)
y_pred = clf.predict(X_test)

#Print the predictions
for i in range(len(test_articles)):
    if y_pred[i] == 1:
        print(f"Article {i+1}: Tsunami")
    else:
        print(f"Article {i+1}: Non-tsunami")