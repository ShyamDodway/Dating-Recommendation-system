# src/features.py
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

def interests_to_corpus(interests_list):
    # Convert list of interests into a space-separated string per user
    return [" ".join([i.replace(" ", "_") for i in lst]) for lst in interests_list]

class InterestVectorizer:
    def __init__(self):
        self.cv = CountVectorizer()

    def fit_transform(self, df: pd.DataFrame):
        corpus = interests_to_corpus(df['Interests'])
        mat = self.cv.fit_transform(corpus)
        return mat.toarray()

    def transform(self, df: pd.DataFrame):
        corpus = interests_to_corpus(df['Interests'])
        mat = self.cv.transform(corpus)
        return mat.toarray()
