import numpy as np
import pandas as pd
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

num_of_recommendations = 20

df_jobs = pd.read_csv('data/20000_Combined_Jobs_Final.csv')[['Job.ID', 'Title', 'Position', 'Company', 'City', 'Job.Description']]
df_jobs = df_jobs.fillna(' ')
df_jobs['text'] = df_jobs['Title'].astype(str) + " " + df_jobs['Position'].astype(str)  + " " + df_jobs['Company'].astype(str) + " " + df_jobs['Job.Description'].astype(str)
df_jobs = df_jobs[['Job.ID', 'text']]
wn = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def clean_txt(text):
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    stop_words = set(stopwords.words('english'))
    clean_text = []
    clean_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if not word in stop_words]
    return " ".join(clean_text)

df_jobs['text'] = df_jobs['text'].apply(clean_txt)

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
job_vec = vectorizer.fit_transform((df_jobs['text']))

with open('interests.txt') as user:
    user_vec = vectorizer.transform(user)    

cos_similarity_vectorizer = map(lambda x: cosine_similarity(user_vec, x), job_vec)
results = list(cos_similarity_vectorizer)

job_matches = {}
df_jobs = pd.read_csv("data/20000_Combined_Jobs_Final.csv")[['Job.ID', 'Title']]
df_ids = df_jobs['Job.ID']

for i, result in enumerate(results):
    job_matches[df_ids.iloc[i]] = result.item()

best_matches = sorted(job_matches.items(), key = lambda x:x[1], reverse=True)[:num_of_recommendations]

for match in best_matches:
    print(str(df_jobs[df_jobs['Job.ID'] == match[0]][['Title']]))
    print(f"match: {match[1]}")