import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from flair.models import TextClassifier
from flair.data import Sentence
from rake_nltk import Rake
import matplotlib.pyplot as plt


# Initialize the Selenium WebDriver
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    driver = webdriver.Chrome(options=options)
    return driver


# Scrape article titles from a news site
def scrape_titles(url):
    driver = init_driver()
    driver.get(url)
    time.sleep(3)  # Wait for the page to load
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    titles = []
    for title in soup.find_all('h2', {'data-testid':'card-headline'}):  # Adjust the tag based on the site's structure
        titles.append(title.get_text(strip=True))

    return titles


# Extract keywords using NLTK-Rake
def extract_keywords(titles):
    r = Rake()
    keywords = []

    for title in titles:
        r.extract_keywords_from_text(title)
        keywords.extend(r.get_ranked_phrases())

    return list(set(keywords))  # Return unique keywords


# Perform sentiment analysis using Flair
def analyze_sentiment(titles):
    classifier = TextClassifier.load('en-sentiment')
    sentiments = []

    for title in titles:
        sentence = Sentence(title)
        classifier.predict(sentence)
        sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'
        print(title, sentence.labels[0])

    return sentiments


# Visualize trends and save the output
def visualize_and_save(sentiments, keywords):
    sentiment_counts = pd.Series(sentiments).value_counts()

    # Plotting the sentiment analysis results
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Sentiment Analysis of Article Titles')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Save the plot
    plt.savefig('sentiment_analysis_results.png')
    plt.close()

    # Save sentiment counts to a CSV file
    sentiment_counts.to_csv('sentiment_counts.csv', header=True)


#Main function to run the pipeline
def main():
    url = 'https://www.bbc.com/news/world/asia/india'  # Replace with actual news site URL
    titles = scrape_titles(url)

    # Store titles in a CSV file
    pd.DataFrame(titles, columns=['Title']).to_csv('article_titles.csv', index=False)

    keywords = extract_keywords(titles)
    print("Extracted Keywords:", keywords)

    sentiments = analyze_sentiment(titles)
    print("Sentiments:", sentiments)

    visualize_and_save(sentiments, keywords)


if __name__ == "__main__":
    main()