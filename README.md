# Dating Matcher

A simple profile matching engine and demo UI for a dating app.  
Features:
- Interest matching using bag-of-words + cosine similarity
- Age, swiping history and relationship preference combined into a single score
- Streamlit demo to explore recommendations
- Modular code for experimentation and extension

## Quick start
1. Create virtualenv and install: `pip install -r requirements.txt`
2. Place dataset at `data/dating_app_dataset.csv` (sample provided)
3. Run demo: `streamlit run app/streamlit_app.py`

## Project structure
app/
- └── streamlit_app.py # Streamlit web app
src/
- ├── data_utils.py # Data loading & cleaning
- ├── recommender.py # Matching engine
- ├── scorer.py # Feature scoring functions
- └── features.py # Interest vectorization
data/
- └── dating_app_dataset.csv # Sample data


##  Run locally
- python -m venv myenv
- myenv\Scripts\activate
- pip install -r requirements.txt
- streamlit run app/streamlit_app.py

## Ideas for improvement
- Replace CountVectorizer with TF-IDF or word embeddings for better interest matching.
- Learn weights with a small labeled dataset using logistic regression.
- Add user cold-start handling and online learning.
- Add A/B test harness and metrics (CTR, matches -> conversations).
