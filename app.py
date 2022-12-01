import streamlit as st
import pandas as pd
import docx2txt
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

st.title("Resume scorer")
# name = st.text_input("Enter your name", '')
# st.write(f"Hello {name}!")
st.subheader("Job description")
uploaded_file = st.file_uploader("Upload job description as a text file (.txt format)")
if uploaded_file is not None:
    job_description = uploaded_file.getvalue()
    job_description = str(job_description)
else:
    st.session_state["upload_state"] = "Upload job description first!"
st.subheader("Resume")
uploaded_resume = st.file_uploader("Upload your resume as a word document (.docx format)")
if uploaded_resume is not None:
#resume = uploaded_resume.getvalue()
    resume = docx2txt.process(uploaded_resume)
    resume = str(resume)
if st.button('Calculate the similarity score between your resume and job description '):
    text = [resume, job_description]
    cv = CountVectorizer(stop_words="english")
    count_matrix = cv.fit_transform(text)
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2) # round to two decimal
    st.write("Your resume matches about "+ str(matchPercentage)+ "% of the job description.")
else:
        st.session_state["upload_state"] = "Upload resume first!"
if st.button('Get the top 3 categories that best suit your resume'):
    stop_words = stopwords.words('english')
    def remove_stop_words (text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
                result.append(token)
        return result
    model_pipeline = load("model_pipeline.joblib")
    def get_category(path):
        #resume = docx2txt.process(path)
        my_resume = docx2txt.process(uploaded_resume)
        my_resume = remove_stop_words(my_resume)
        my_resume = pd.Series(" ".join(my_resume))
        probs = model_pipeline.predict_proba(my_resume)[0]
        rf = model_pipeline['randomforestclassifier']
        return pd.DataFrame({"Category":rf.classes_, "prob":probs}).sort_values("prob", ascending=False, ignore_index= True).head(3)
    result = get_category(resume)
    st.write("The top 3 categories that best suits your resume are:")
    st.dataframe(result)




