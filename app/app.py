import scikitlearn
import streamlit as st
import numpy as np  # give array
import pandas as pd # data clean and get data to help
import re     # regural give and make pattern
from nltk.corpus import stopwords # stopword -->like the ,for,of,in,with
import nltk
from nltk.stem import PorterStemmer
#from nltk.stem.porter import PorterStremmer # PorterStremmer --> charge word or convert base word example loved ,loving == love (chnage base word)`
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer -->convert word into Array ex loved =[0.0]
from sklearn.model_selection import train_test_split #train_test_split -->data split
from sklearn.linear_model import LogisticRegression #LogisricRegression -->use classifiction problem
from sklearn.metrics import accuracy_score # check model accuracy 

dataset = pd.read_csv('train.csv')
dataset.head()
dataset.shape
dataset.isna().sum()
dataset['title']
# Preprocessing function
def preprocess_title(title):
    title = re.sub(r'\W', ' ', title)  # Remove special characters
    title = title.lower()  # Convert to lowercase
    title = title.split()  # Tokenize
    title = [word for word in title if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(title)
# Apply preprocessing
dataset['title'] = dataset['title'].apply(preprocess_title)

vector = TfidfVectorizer()
vector.fit(X)
X =vector.transform(X)
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1)

#LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

#website
st.title('Fake News Detector')
input_text =st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')