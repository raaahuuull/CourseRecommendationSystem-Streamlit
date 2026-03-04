# ==================================================
# COURSE RECOMMENDATION SYSTEM
# ==================================================

import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Course Recommendation System")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    with open("courses_data.pkl", "rb") as f:
        df = pickle.load(f)

    df = df.reset_index(drop=True)
    return df

df = load_data()

# --------------------------------------------------
# Build similarity model
# --------------------------------------------------
@st.cache_resource
def build_model(df):

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["course_name"])

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)

    return tfidf_matrix, model

tfidf_matrix, model = build_model(df)

# --------------------------------------------------
# Recommendation function
# --------------------------------------------------
def recommend_courses(selected_course, top_n=5):

    idx = df[df["course_name"] == selected_course].index[0]

    distances, indices = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=min(top_n + 15, len(df))
    )

    neighbors = indices.flatten()[1:]
    sim_scores = 1 - distances.flatten()[1:]

    rec_df = df.iloc[neighbors].copy()
    rec_df["similarity"] = sim_scores

    # normalize rating and price
    scaler = MinMaxScaler()
    rec_df[['rating_norm','price_norm']] = scaler.fit_transform(
        rec_df[['rating','course_price']]
    )

    rec_df["price_score"] = 1 - rec_df["price_norm"]

    # similarity has highest weight
    rec_df["final_score"] = (
        0.6 * rec_df["similarity"] +
        0.3 * rec_df["rating_norm"] +
        0.1 * rec_df["price_score"]
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

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🎓 Course Recommendation System")

selected_course = st.selectbox(
    "Select a Course",
    sorted(df["course_name"].unique())
)

top_n = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Show Recommendations"):
    recommendations = recommend_courses(selected_course, top_n)
    st.subheader("Recommended Courses")
    st.dataframe(recommendations)
