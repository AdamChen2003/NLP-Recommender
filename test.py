import numpy as np
import pandas as pd
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

df_jobs = pd.read_csv('data/Combined_Jobs_Final.csv')[['Job.ID', 'Title', 'Position', 'Company', 'City', 'Job.Description']].iloc[:5000]
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

df_job_view = pd.read_csv('data/Job_Views.csv')[['Applicant.ID', 'Job.ID', 'Position', 'Company', 'City']]
df_job_view['select_pos_com_city'] = df_job_view['Position'].astype(str) + ' ' + df_job_view['Company'].astype(str) + ' ' + df_job_view['City'].astype(str)
df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].apply(clean_txt)
df_job_view = df_job_view[['Applicant.ID', 'select_pos_com_city']]

df_experience = pd.read_csv('data/Experience.csv')[['Applicant.ID', 'Position.Name']]
df_experience['Position.Name'] = df_experience['Position.Name'].map(str).apply(clean_txt)

df_poi = pd.read_csv('Data/Positions_Of_Interest.csv', sep=',')[['Applicant.ID', 'Position.Of.Interest']]
df_poi = df_poi.sort_values(by='Applicant.ID')
df_poi['Position.Of.Interest'] = df_poi['Position.Of.Interest'].map(str).apply(clean_txt)
df_poi = df_poi.fillna(' ')

df_final_person = pd.merge(pd.merge(df_job_view, df_experience, on='Applicant.ID'), df_poi, on='Applicant.ID')
df_final_person['text'] = df_final_person['select_pos_com_city'] + ' ' + df_final_person['Position.Name'] + ' ' + df_final_person['Position.Of.Interest']
df_final_person['text'] = df_final_person['text'].map(str).apply(clean_txt)
df_final_person = df_final_person[['Applicant.ID', 'text']]

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
job_vec = vectorizer.fit_transform((df_jobs['text']))

user_vec = vectorizer.transform(df_final_person['text'])
cos_similarity_vectorizer = map(lambda x: cosine_similarity(user_vec, x), job_vec)
results = list(cos_similarity_vectorizer)

job_matches = {}
df_jobs = pd.read_csv("data/Combined_Jobs_Final.csv")[['Job.ID', 'Title', 'Position', 'Company', 'City', 'Job.Description']].iloc[:5000]
df_ids = df_jobs['Job.ID']

user_index = 100
print(str(df_final_person.iloc[user_index]['text']))

for i, result in enumerate(results):
    job_matches[df_ids.iloc[i]] = result[user_index].item()

best_matches = sorted(job_matches.items(), key = lambda x:x[1], reverse=True)[:10]

for match in best_matches:
    print(str(df_jobs[df_jobs['Job.ID'] == match[0]][['Title']]))
    print(match[1])