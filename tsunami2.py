import pandas as pd
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Scrape news articles related to tsunamis
tsunami_articles = []

urls = ['https://www.reuters.com/world/asia-pacific/earthquake-with-61-magnitude-hits-northern-japan-authorities-say-2023-02-25/',
        'https://www.ndtv.com/world-news/magnitude-6-1-earthquake-shakes-japans-hokkaido-no-tsunami-warning-3814273',
        'https://apnews.com/article/turkey-japan-earthquakes-4547c9e6d98bc8cfecd2fc0c99fa1a83',
        'https://www.hindustantimes.com/world-news/earthquake-of-magnitude-6-1-jolts-japan-s-hokkaido-no-tsunami-warning-101677334524481.html',
        'https://apnews.com/article/turkey-japan-earthquakes-4547c9e6d98bc8cfecd2fc0c99fa1a83']

for url in urls:
    article = Article(url)
    article.download()
    article.parse()
    tsunami_articles.append(article.text)

# Scrape non-tsunami news articles
non_tsunami_articles = []

urls = ['https://www.cnn.com/2023/03/05/politics/eric-adams-lori-lightfoot-crime-cnntv/index.html',
        'https://nypost.com/2023/03/05/adams-says-lightfoots-loss-in-chicago-is-warning-sign-for-country/',
        'https://abcnews.go.com/International/wireStory/thai-cave-boys-ashes-arrive-home-uk-final-97627228',
        'https://nypost.com/article/best-nordstrom-gifts/',    
        'https://www.theatlantic.com/magazine/archive/2022/07/last-resort-beach-vacation-environmental-impact/638448/',
        'https://www.nbcnews.com/better/health/what-beach-does-your-brain-ncna787231',
    ]

for url in urls:
    article = Article(url)
    article.download()
    article.parse()
    non_tsunami_articles.append(article.text)

# Create a Pandas DataFrame
df = pd.DataFrame({'text': tsunami_articles + non_tsunami_articles, 'tsunami': [1]*len(tsunami_articles) + [0]*len(non_tsunami_articles)})

# Preprocess the data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Train the Naive Bayes classifier
y = df['tsunami']
clf = MultinomialNB()
clf.fit(X, y)

# Test the classifier
test_articles = []

test_urls = ['https://news.delta.com/deltas-statement-lifting-international-testing-requirement-us-entry',
             'https://www.latimes.com/california/story/2023-03-05/search-continues-for-gunman-responsible-for-shooting-that-left-5-wounded-at-san-pedro-beach',
            'https://www.technologynetworks.com/applied-sciences/news/devastating-puerto-rico-1918-tsunami-wasnt-caused-by-a-landslide-368762',
            'https://www.staradvertiser.com/2023/01/08/breaking-news/no-tsunami-threat-to-hawaii-after-strong-quake-hits-vanuatu/',
            'https://www.staradvertiser.com/2023/03/05/sports/sports-breaking/jon-jones-returns-to-win-ufc-heavyweight-title-in-1st-round/',
            'https://www.foxnews.com/us/california-suspects-sought-5-shot-los-angeles-area-beach']

for url in test_urls:
    article = Article(url)
    article.download()
    article.parse()
    test_articles.append(article.text)

X_test = vectorizer.transform(test_articles)
y_pred = clf.predict(X_test)

#Print the predictions
for i in range(len(test_articles)):
    if y_pred[i] == 1:
        print(f"Article {i+1}: Tsunami")
    else:
        print(f"Article {i+1}: Non-tsunami")