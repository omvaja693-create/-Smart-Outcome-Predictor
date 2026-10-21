# 🎯 Smart Outcome Predictor

**An ensemble machine learning system that predicts online course completion and final student performance.**

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Coming%20Soon-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> 🏫 Red & White Skill Education, Project PR. 5

---

## 🔗 Live Demo

🚀 **Live App:** `https://your-app-name.streamlit.app` *(coming soon, to be deployed on Streamlit)*

---

## 📌 Overview

Smart Outcome Predictor uses student demographics and learning-platform engagement data to solve two problems at once:

| Task | Target | Type |
|------|--------|------|
| ✅ Course completion | `completion_status` (0 or 1) | Classification |
| 📊 Final performance | `final_score` (0 to 100) | Regression |

The project compares single models against ensemble methods and exports the best models for deployment.

---

## ✨ Features

- 🌳 Single base models (Decision Tree) as the benchmark
- 🧺 Bagging (Bootstrap Aggregating)
- 🚀 Boosting with AdaBoost and XGBoost
- 🗳️ Voting ensembles (hard and soft)
- 🧱 Stacking ensembles with a meta-model
- 📈 Side-by-side evaluation for classification and regression
- 💾 Model and metadata export, ready for deployment

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **Modeling:** scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Deployment (planned):** Streamlit

---

## 📂 Dataset

`Smart_Outcome_Predictor_Dataset_5200.xlsx` holds **5,200 student records** with 19 original columns.

| Group | Columns |
|-------|---------|
| 👤 Demographics | `age`, `country_region`, `education_background` |
| 💻 Platform | `device_type`, `course_level`, `course_category`, `course_start_date`, `week_of_year` |
| 📚 Engagement | `sessions`, `time_spent_hours`, `videos_watched`, `quiz_attempts`, `assignments_submitted`, `forum_posts`, `avg_quiz_score`, `attendance_rate` |
| 🎯 Targets | `completion_status`, `final_score` |

---

## 🔬 Methodology

1. **Cleaning:** fill missing numeric values with the median, drop `student_id` and `course_start_date`
2. **Encoding:** one-hot encoding for categorical columns (`drop_first=True`)
3. **Split:** 80/20 train-test split (stratified for classification)
4. **Scaling:** `StandardScaler`
5. **Modeling:** train and compare all ensemble techniques
6. **Evaluation:** compare metrics across all models
7. **Export:** save the best models and metadata

### 🤖 Models Compared

| Technique | Classification | Regression |
|-----------|----------------|------------|
| Single model | Decision Tree | Decision Tree |
| Bagging | BaggingClassifier | BaggingRegressor |
| Boosting | AdaBoost, XGBoost | AdaBoost, XGBoost |
| Voting | Hard and Soft Voting | n/a |
| Stacking | Stacking (meta: Logistic Regression) | Stacking (meta: Ridge) |

### 📏 Metrics

- **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression:** MAE, RMSE, R² Score

---

## 🏆 Results

Add your final benchmark tables here after running the notebook.

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Stacking Classifier | – | – | – |

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| XGBoost Regressor | – | – | – |

**Selected models:** Stacking Classifier (classification) and XGBRegressor (regression)

---

## 📁 Project Structure

```
Smart_Outcome_Predictor/
├── data/
│   └── Smart_Outcome_Predictor_Dataset_5200.xlsx
├── notebooks/
│   └── Smart_Outcome_Predictor_Updated.ipynb
├── model/
│   ├── smart_outcome_predictor.pkl
│   └── model_metadata.json
├── app.py                # Streamlit app (coming soon)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Smart_Outcome_Predictor.git
cd Smart_Outcome_Predictor

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl jupyter
```

---

## ▶️ Usage

**1. Train the models**

Open and run all cells in `notebooks/Smart_Outcome_Predictor_Updated.ipynb`. The trained artifacts are saved to the `model/` folder.

**2. Load the saved model**

```python
import pickle
import pandas as pd

with open("model/smart_outcome_predictor.pkl", "rb") as f:
    artifacts = pickle.load(f)

classifier = artifacts["classifier"]
regressor = artifacts["regressor"]
feature_names = artifacts["feature_names"]

# Build a one-row DataFrame with the same feature columns
sample = pd.DataFrame([your_input_dict])[feature_names]

print("Completion:", classifier.predict(sample)[0])
print("Final score:", regressor.predict(sample)[0])
```

---

## 🗺️ Roadmap

- [x] Data preparation and encoding
- [x] Bagging, Boosting, Voting, and Stacking
- [x] Model evaluation and comparison
- [x] Model and metadata export
- [ ] Streamlit web app
- [ ] Live deployment

---

## 👤 Author

**Mr Om**
Data Analyst | Statistics • Python • Data Analysis

---

⭐ If you found this project useful, consider giving it a star.
