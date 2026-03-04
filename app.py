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


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    with open("courses_data.pkl", "rb") as f:
        df = pickle.load(f)

    df = df.reset_index(drop=True)

    return df


df = load_data()


# ==================================================
# BUILD SIMILARITY MODEL
# ==================================================

@st.cache_resource
def build_model(df):

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(df["course_name"])

    model = NearestNeighbors(metric="cosine", algorithm="brute")

    model.fit(tfidf_matrix)

    return tfidf_matrix, model


tfidf_matrix, model = build_model(df)


# ==================================================
# RECOMMENDATION FUNCTION
# ==================================================

def recommend_courses(selected_course, top_n=5):

    # -----------------------------
    # STEP 1: keyword-based related courses
    # -----------------------------

    keywords = selected_course.lower().split()

    related = df[df["course_name"].str.lower().apply(
        lambda x: any(word in x for word in keywords)
    )]

    related = related[related["course_name"] != selected_course]

    if len(related) >= top_n:

        return related.head(top_n)[[
            "course_id",
            "course_name",
            "difficulty_level",
            "rating",
            "course_price"
        ]]

    # -----------------------------
    # STEP 2: similarity model
    # -----------------------------

    idx = df[df["course_name"] == selected_course].index[0]

    distances, indices = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=min(top_n + 10, len(df))
    )

    neighbors = indices.flatten()[1:]

    rec_df = df.iloc[neighbors].copy()

    # ranking logic
    scaler = MinMaxScaler()

    rec_df[['rating_norm','price_norm']] = scaler.fit_transform(
        rec_df[['rating','course_price']]
    )

    rec_df["price_score"] = 1 - rec_df["price_norm"]

    rec_df["final_score"] = (
        0.7 * rec_df["rating_norm"] +
        0.3 * rec_df["price_score"]
    )

    rec_df = rec_df.sort_values("final_score", ascending=False)

    # -----------------------------
    # STEP 3: fallback top courses
    # -----------------------------

    if len(rec_df) < top_n:

        fallback = df.sort_values("rating", ascending=False)

        rec_df = pd.concat([rec_df, fallback])

    result = rec_df.drop_duplicates("course_name").head(top_n)[[
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

top_n = st.slider("Number of Recommendations", 1, 10, 5)


if st.button("Show Recommendations"):

    recommendations = recommend_courses(selected_course, top_n)

    st.subheader("Recommended Courses")

    st.dataframe(recommendations)
