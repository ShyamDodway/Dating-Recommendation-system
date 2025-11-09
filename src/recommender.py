# src/recommender.py
import numpy as np
import pandas as pd
from .features import InterestVectorizer
from .scorer import combined_score

class Recommender:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.vec = InterestVectorizer()
        self.interest_matrix = self.vec.fit_transform(self.df)

    def recommend_for_user(self, user_id, top_k=5, gender_filter=None):
        # Find user row
        user_row = self.df[self.df['User ID'] == user_id]
        if user_row.empty:
            raise ValueError(f"User {user_id} not found.")
        user_idx = user_row.index[0]
        user = self.df.loc[user_idx]
        user_vec = self.interest_matrix[user_idx]

        candidates = self.df
        if gender_filter:
            candidates = candidates[candidates['Gender'] == gender_filter]

        results = []
        for idx, candidate in candidates.iterrows():
            if candidate['User ID'] == user_id:
                continue
            cand_vec = self.interest_matrix[idx]
            score = combined_score(
                user_vec, cand_vec,
                user['Age'], candidate['Age'],
                user['Swiping History'], candidate['Swiping History'],
                user['Looking For'], candidate['Looking For']
            )
            results.append((candidate['User ID'], float(score), candidate))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def all_recommendations(self, gender_from='Male', gender_to='Female', top_k=1):
        from_group = self.df[self.df['Gender'] == gender_from]
        recs = []
        for _, user in from_group.iterrows():
            user_id = user['User ID']
            res = self.recommend_for_user(user_id, top_k=top_k, gender_filter=gender_to)
            recs.append((user_id, res))
        return recs
