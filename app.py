import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler



@st.cache_data
def load_data():
    with open("courses_data.pkl", "rb") as f:
        df = pickle.load(f)

    df = df.reset_index(drop=True)
    return df


df = load_data()



def recommend_courses(selected_course, top_n=10):

    selected_row = df[df["course_name"] == selected_course].iloc[0]
    selected_topic = selected_row["topic"]

    

    related = df[df["topic"] == selected_topic].copy()
    related = related[related["course_name"] != selected_course]

    if len(related) > 0:

        scaler = MinMaxScaler()

        related[['rating_norm','price_norm']] = scaler.fit_transform(
            related[['rating','course_price']]
        )

        related["price_score"] = 1 - related["price_norm"]

        related["final_score"] = (
            0.7 * related["rating_norm"] +
            0.3 * related["price_score"]
        )

        related = related.sort_values("final_score", ascending=False)

 

    top_courses = df[df["course_name"] != selected_course].copy()
    top_courses = top_courses.sort_values("rating", ascending=False)


    top_courses = top_courses[
        ~top_courses["course_name"].isin(related["course_name"])
    ]

  

    related_courses = related.head(top_n)

    remaining = top_n - len(related_courses)

    top_courses = top_courses.head(remaining)

    return related_courses, top_courses




st.title("🎓 Course Recommendation System")

selected_course = st.selectbox(
    "Select a Course",
    sorted(df["course_name"].unique())
)

top_n = st.slider(
    "Number of Recommendations",
    5,
    10,
    8
)

if st.button("Show Recommendations"):

    related_courses, top_courses = recommend_courses(selected_course, top_n)

  

    if len(related_courses) > 0:

        st.success("Showing related courses")

        st.subheader(" Related Courses")

        st.dataframe(
            related_courses[[
                "course_id",
                "course_name",
                "difficulty_level",
                "rating",
                "course_price"
            ]]
        )

    else:
        st.warning("No related courses found.")

   

    if len(top_courses) > 0:

        st.subheader(" Top Rated Courses")

        st.dataframe(
            top_courses[[
                "course_id",
                "course_name",
                "difficulty_level",
                "rating",
                "course_price"
            ]]
        )

