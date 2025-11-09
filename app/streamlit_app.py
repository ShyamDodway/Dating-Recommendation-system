# app/streamlit_app.py
import sys, os
# 🔧 Add project root to sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.data_utils import load_data
from src.recommender import Recommender

st.set_page_config(page_title="Dating Matcher", layout="centered")

st.title("Dating Matcher — Demo")

DATA_PATH = st.sidebar.text_input("Path to CSV", "Data/dating_app_dataset.csv")
top_k = st.sidebar.slider("Top K matches", 1, 10, 5)

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.success(f"Loaded {len(df)} profiles")

user_ids = df['User ID'].tolist()
selected_user = st.selectbox("Pick a user to find matches for", user_ids)

gender_filter = st.radio("Search for gender", options=["Female", "Male"], index=0)

if st.button("Find matches"):
    rec = Recommender(df)
    matches = rec.recommend_for_user(selected_user, top_k=top_k, gender_filter=gender_filter)
    if not matches:
        st.info("No matches found.")
    else:
        for uid, score, candidate in matches:
            st.write(f"**User {uid}** — Score: {score:.3f}")
            st.write(f"Age: {candidate['Age']}, Interests: {candidate['Interests']}, Looking For: {candidate['Looking For']}")
            st.markdown("---")
