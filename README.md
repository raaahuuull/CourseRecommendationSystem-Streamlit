#  Course Recommendation System (Streamlit)

[![Python](https://img.shields.io/badge/Python-3.9-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Recommendation%20System-green)]()

 **Live Demo:**  
https://courserecommendationsystem-app-8qvyhxcjz8jgpxzheeyzbl.streamlit.app/

Built using **Python, Machine Learning, and Streamlit**, this project recommends relevant online courses based on **topic similarity, course ratings, and price optimization**.

---

#  Project Overview

Online learning platforms contain thousands of courses, making it difficult for learners to identify the most relevant ones.

This project builds a **Course Recommendation System** that suggests courses based on:

- Topic similarity
- Course ratings
- Course price

If there are not enough related courses available, the system fills the remaining recommendations with **top-rated courses from the dataset**.

---

#  Features

- Interactive **Streamlit Web Application**
- Select any course from the dataset
- Adjustable **number of recommendations**
- Displays two types of recommendations:
  - Related Courses
  - Top Rated Courses
- Ranking system using **weighted scoring**
- Fast loading using **Streamlit caching**

---

#  Recommendation Logic

The recommendation system works in the following steps:

###  Find Related Courses

Courses belonging to the **same topic** as the selected course are filtered.

###  Normalize Important Features

Two important features are normalized using **MinMaxScaler**

- Rating
- Course Price

###  Score Calculation

Courses are ranked using the formula:

```
Final Score = (0.7 × Rating Score) + (0.3 × Price Score)
```

Where:

- Higher ratings increase the score
- Lower prices increase the score

###  Generate Recommendations

1. Top courses from the same topic are selected.
2. If fewer courses exist, the remaining slots are filled with **top rated courses** from the dataset.

---

# 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

---

# Dataset

The project uses an **Online Course Recommendation Dataset** containing course details such as:

| Feature | Description |
|------|------|
| user_id | Unique identifier for learners |
| course_id | Unique course identifier |
| course_name | Name of the course |
| instructor | Course instructor |
| course_duration_hours | Course duration |
| certification_offered | Certification availability |
| difficulty_level | Beginner / Intermediate / Advanced |
| rating | Course rating (1–5) |
| enrollment_numbers | Number of enrolled students |
| course_price | Price of the course |
| feedback_score | Sentiment feedback score |
| study_material_available | Availability of study material |
| time_spent_hours | Average time spent |
| previous_courses_taken | Number of previous courses |

---

#  Project Structure

```
CourseRecommendationSystem-Streamlit
│
├── notebooks
│   └── CRS.ipynb
│
├── screenshots
│   ├── interface.png
│   └── recommendations.png
│
├── app.py
├── courses_data.pkl
├── requirements.txt
└── README.md
```

### File Description

**app.py**

Main Streamlit application that runs the recommendation system.

**notebooks/CRS.ipynb**

Jupyter Notebook used for:

- Data exploration
- Data preprocessing
- Feature engineering
- Saving the processed dataset

**courses_data.pkl**

Serialized dataset used by the Streamlit application.

**screenshots**

Contains images of the application interface and recommendation output.

---

#  Installation

Clone the repository

```bash
git clone https://github.com/raaahuull/CourseRecommendationSystem-Streamlit.git
cd CourseRecommendationSystem-Streamlit
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

Run the Streamlit application

```bash
streamlit run app.py
```

Open in browser

```
http://localhost:8501
```

---

# Application Screenshots

### Course Selection Interface

Users select a course and choose the number of recommendations.

### Recommendation Output

The system displays:

- Related Courses
- Top Rated Courses

Each recommendation includes:

- Course ID
- Course Name
- Difficulty Level
- Rating
- Course Price

---

# Live Application

You can try the deployed application here:

https://courserecommendationsystem-app-8qvyhxcjz8jgpxzheeyzbl.streamlit.app/

---

# Future Improvements

- Content-based filtering using course descriptions
- Collaborative filtering using user behavior
- Deep learning based recommendation systems
- Personalized recommendations
- Deployment using Docker / Cloud platforms

---

# License

This project is for **educational and learning purposes**.

---

# Author

**Rahul Raj**  
Aspiring Data Scientist | Machine Learning Enthusiast

- GitHub: https://github.com/raaahuull
- LinkedIn: https://www.linkedin.com/in/rahul-raj-ds

---

Feel free to explore the project and give it a star if you like it!
