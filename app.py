"""Smart Outcome Predictor - simple CLI app.

Asks for student info, then predicts:
  - course completion (classification)
  - final score (regression)

Run:  python app.py
"""
import os
import pickle
import warnings

import pandas as pd

warnings.filterwarnings("ignore")  # hide sklearn version-mismatch noise

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "smart_outcome_predictor.pkl")

# Categorical options. First option = dropped baseline in training (drop_first=True).
CHOICES = {
    "country_region": ["Africa", "Americas", "Asia", "Europe", "Oceania"],
    "device_type": ["Laptop", "Mobile", "Tablet"],
    "education_background": ["Graduate", "HighSchool", "Undergrad", "WorkingPro"],
    "course_level": ["Advanced", "Beginner", "Intermediate"],
    "course_category": ["Business", "Data", "Design", "Marketing", "Programming"],
}

# (prompt, column, type, min, max)
NUMERIC = [
    ("Age", "age", int, 16, 45),
    ("Week of year", "week_of_year", int, 1, 52),
    ("Sessions", "sessions", int, 1, 94),
    ("Time spent (hours)", "time_spent_hours", float, 0.6, 76.8),
    ("Videos watched", "videos_watched", int, 0, 158),
    ("Quiz attempts", "quiz_attempts", int, 0, 20),
    ("Assignments submitted", "assignments_submitted", int, 0, 12),
    ("Forum posts", "forum_posts", int, 0, 8),
    ("Average quiz score (0-100)", "avg_quiz_score", float, 0, 100),
    ("Attendance rate (0-1)", "attendance_rate", float, 0.0, 1.0),
]


def ask_number(label, kind, lo, hi):
    while True:
        raw = input(f"{label} [{lo}-{hi}]: ").strip()
        try:
            val = kind(raw)
        except ValueError:
            print("  Enter a valid number.")
            continue
        if not lo <= val <= hi:
            print(f"  Must be between {lo} and {hi}.")
            continue
        return val


def ask_choice(label, options):
    print(f"{label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("Pick number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice.")


def build_row(answers, feature_names):
    """One-hot encode answers into the exact training feature layout."""
    row = {f: 0 for f in feature_names}
    for col, val in answers.items():
        if col in row:
            row[col] = val
        else:  # categorical -> column like 'device_type_Mobile' (baseline has none)
            dummy = f"{col}_{val}"
            if dummy in row:
                row[dummy] = 1
    return pd.DataFrame([row], columns=feature_names).astype(float)


def main():
    with open(MODEL_PATH, "rb") as f:
        art = pickle.load(f)
    clf, reg, names = art["classifier"], art["regressor"], art["feature_names"]

    print("=== Smart Outcome Predictor ===\n")
    while True:
        answers = {}
        for label, col, kind, lo, hi in NUMERIC:
            answers[col] = ask_number(label, kind, lo, hi)
        for col, opts in CHOICES.items():
            answers[col] = ask_choice(col.replace("_", " ").title(), opts)

        X = build_row(answers, names)  # models were trained on unscaled features
        prob = clf.predict_proba(X)[0][1]
        completed = int(clf.predict(X)[0])
        score = max(0.0, min(100.0, float(reg.predict(X)[0])))

        print("\n--- Prediction ---")
        print(f"Completion : {'WILL COMPLETE' if completed else 'WILL NOT COMPLETE'} "
              f"(probability {prob:.1%})")
        print(f"Final score: {score:.1f} / 100\n")

        if input("Predict another? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
