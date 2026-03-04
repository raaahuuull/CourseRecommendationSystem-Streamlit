# ==================================================
# COURSE RECOMMENDATION SYSTEM - STREAMLIT APP
# ==================================================

import streamlit as st
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Course Recommendation System")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    with open("courses_data.pkl", "rb") as f:
        df = pickle.load(f)
    return df


df = load_data()

# ==================================================
# BUILD TEXT SIMILARITY MODEL
# ==================================================

@st.cache_resource
def build_model(df):

    text_data = df["course_name"] + " " + df["topic"]

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(text_data)

    model = NearestNeighbors(metric="cosine", algorithm="brute")

    model.fit(tfidf_matrix)

    return tfidf_matrix, model


tfidf_matrix, model = build_model(df)

# ==================================================
# RECOMMENDATION FUNCTION
# ==================================================

def recommend_courses(selected_course, top_n=5):

    idx = df[df["course_name"] == selected_course].index[0]

    distances, indices = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=min(top_n + 10, len(df))
    )

    course_indices = indices.flatten()[1:]

    rec_df = df.iloc[course_indices].copy()

    # Ranking logic
    scaler = MinMaxScaler()

    rec_df[['rating_norm','price_norm']] = scaler.fit_transform(
        rec_df[['rating','course_price']]
    )

    rec_df["price_score"] = 1 - rec_df["price_norm"]

    rec_df["final_score"] = (
        0.7 * rec_df["rating_norm"] +
        0.3 * rec_df["price_score"]
    )

    result = rec_df.sort_values(
        "final_score",
        ascending=False
    ).head(top_n)[[
        "course_id",
        "course_name",
        "difficulty_level",
        "rating",
        "course_price"
    ]]

    return result


# ==================================================
# STREAMLIT UI
# ==================================================

st.title("🎓 Course Recommendation System")

selected_course = st.selectbox(
    "Select a Course",
    sorted(df["course_name"].unique())
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Show Recommendations"):

    recommendations = recommend_courses(selected_course, top_n)

    st.subheader("Recommended Courses")

    st.dataframe(recommendations)
