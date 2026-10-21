"""Smart Outcome Predictor - Streamlit app.

Single page: enter student info, click Predict, see:
  - course completion (classification)
  - final score (regression)

Run locally:  streamlit run app.py
"""
import os
import pickle
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "smart_outcome_predictor.pkl")

# First option of each list = dropped baseline in training (drop_first=True).
CHOICES = {
    "country_region": ("Country / Region", ["Africa", "Americas", "Asia", "Europe", "Oceania"]),
    "device_type": ("Device", ["Laptop", "Mobile", "Tablet"]),
    "education_background": ("Education", ["Graduate", "HighSchool", "Undergrad", "WorkingPro"]),
    "course_level": ("Course level", ["Advanced", "Beginner", "Intermediate"]),
    "course_category": ("Course category", ["Business", "Data", "Design", "Marketing", "Programming"]),
}


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_row(answers, feature_names):
    """One-hot encode answers into exact training feature layout."""
    row = {f: 0 for f in feature_names}
    for col, val in answers.items():
        if col in row:
            row[col] = val
        else:
            dummy = f"{col}_{val}"
            if dummy in row:
                row[dummy] = 1
    return pd.DataFrame([row], columns=feature_names).astype(float)


st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎯")
st.title("🎯 Smart Outcome Predictor")
st.caption("Enter student details to predict course completion and final score.")

art = load_artifacts()
clf, reg, names = art["classifier"], art["regressor"], art["feature_names"]

with st.form("predict_form"):
    st.subheader("Student profile")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 16, 45, 25)
        week = st.number_input("Week of year", 1, 52, 12)
        country = st.selectbox(CHOICES["country_region"][0], CHOICES["country_region"][1])
        education = st.selectbox(CHOICES["education_background"][0], CHOICES["education_background"][1])
        device = st.selectbox(CHOICES["device_type"][0], CHOICES["device_type"][1])
    with c2:
        level = st.selectbox(CHOICES["course_level"][0], CHOICES["course_level"][1])
        category = st.selectbox(CHOICES["course_category"][0], CHOICES["course_category"][1])

    st.subheader("Engagement")
    c3, c4 = st.columns(2)
    with c3:
        sessions = st.number_input("Sessions", 1, 94, 20)
        hours = st.number_input("Time spent (hours)", 0.6, 76.8, 15.0, step=0.5)
        videos = st.number_input("Videos watched", 0, 158, 30)
        quizzes = st.number_input("Quiz attempts", 0, 20, 5)
    with c4:
        assignments = st.number_input("Assignments submitted", 0, 12, 4)
        forum = st.number_input("Forum posts", 0, 8, 1)
        quiz_score = st.number_input("Average quiz score (0-100)", 0.0, 100.0, 60.0, step=1.0)
        attendance = st.number_input("Attendance rate (0-1)", 0.0, 1.0, 0.7, step=0.05)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    answers = {
        "age": age, "week_of_year": week, "sessions": sessions,
        "time_spent_hours": hours, "videos_watched": videos,
        "quiz_attempts": quizzes, "assignments_submitted": assignments,
        "forum_posts": forum, "avg_quiz_score": quiz_score,
        "attendance_rate": attendance,
        "country_region": country, "device_type": device,
        "education_background": education, "course_level": level,
        "course_category": category,
    }
    X = build_row(answers, names)
    prob = float(clf.predict_proba(X)[0][1])
    completed = int(clf.predict(X)[0])
    score = max(0.0, min(100.0, float(reg.predict(X)[0])))

    st.subheader("Prediction")
    r1, r2 = st.columns(2)
    r1.metric("Completion", "Will complete" if completed else "Will not complete",
              f"{prob:.1%} probability", delta_color="off")
    r2.metric("Final score", f"{score:.1f} / 100")
    if completed:
        st.success("Student likely completes the course.")
    else:
        st.warning("Student at risk of not completing.")
