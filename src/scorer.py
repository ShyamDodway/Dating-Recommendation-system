# src/scorer.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def age_score(a1, a2, max_span=20):
    # Score in [0,1] where closer ages -> higher score
    diff = abs(a1 - a2)
    s = max(0, (max_span - diff) / max_span)
    return s

def swipe_score(s1, s2):
    # Use min of the two normalized (0..1)
    return min(s1, s2)/100.0

def relationship_score(r1, r2):
    return 1.0 if r1 == r2 else 0.0

def combined_score(interest_vec1, interest_vec2, age1, age2, swipe1, swipe2, rel1, rel2, weights=None):
    """
    Combine:
     - interest similarity (cosine)
     - age_score
     - swipe_score
     - relationship_score
    weights: dict of weights
    """
    if weights is None:
        weights = {'interest':0.5, 'age':0.2, 'swipe':0.2, 'rel':0.1}

    # interest similarity
    interest_sim = float(cosine_similarity([interest_vec1], [interest_vec2])[0][0])
    a = age_score(age1, age2)
    s = swipe_score(swipe1, swipe2)
    r = relationship_score(rel1, rel2)

    # Combine using weights
    total = weights['interest']*interest_sim + weights['age']*a + weights['swipe']*s + weights['rel']*r
    return total
