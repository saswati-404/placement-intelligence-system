# 🎓 Placement Intelligence System

An AI-powered student placement prediction system built with Python, Scikit-learn, and Streamlit. Predicts placement likelihood, estimates salary range, and delivers personalized career recommendations.

---

## ✨ Features

- **Placement Prediction** — ML model trained on 500 student records (85% accuracy)
- **Probability Score** — Confidence percentage for placement outcome
- **Smart Recommendations** — 7-factor personalized action plan
- **EDA Dashboard** — 6 interactive visualizations with key insights
- **Feature Importance** — Understand what actually drives placements

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data  | Pandas, NumPy |
| ML    | Scikit-learn (Logistic Regression, Random Forest) |
| Viz   | Matplotlib, Seaborn |
| App   | Streamlit |
| Save  | Joblib |

---

## 📁 Project Structure

```
placement-intelligence-system/
│
├── data/
│   ├── placement_data.csv        ← Dataset (500 students)
│   └── generate_data.py          ← Data generation script
│
├── notebooks/
│   ├── eda_and_ml.py             ← EDA + ML training pipeline
│   └── plots/                    ← Generated charts
│       ├── eda_dashboard.png
│       ├── correlation_heatmap.png
│       └── feature_importance.png
│
├── app/
│   └── app.py                    ← Streamlit application
│
├── models/
│   ├── placement_model.pkl       ← Trained ML model
│   ├── scaler.pkl                ← Feature scaler
│   ├── label_encoder.pkl         ← Encoder for stream
│   └── features.pkl              ← Feature list
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/placement-intelligence-system.git
cd placement-intelligence-system
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Generate dataset & train models
```bash
cd data
python generate_data.py
cd ../notebooks
python eda_and_ml.py
```

### Step 4 — Launch the app
```bash
cd ../app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **85%** ✅ |
| Random Forest | 82% |

---

## 🔑 Key Insights

- Placed students have avg CGPA of **7.46** vs **6.67** for non-placed
- Students with 2+ internships have **91.9%** placement rate vs **39.4%** with none
- Backlogs significantly reduce placement chances

---

## 💼 Resume Line

> Developed a Placement Intelligence System using Python, Pandas, and Scikit-learn to analyze student data and predict placement outcomes with 85% accuracy. Built an interactive Streamlit dashboard with personalized career recommendations based on 8 student performance features.

---

## 👤 Author

Built as a data science portfolio project.
