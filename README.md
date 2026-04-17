# 🎓 Placement Intelligence System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![GitHub stars](https://img.shields.io/github/stars/saswati-404/placement-intelligence-system?style=social)

---

## 🚀 Overview

An AI-powered system that predicts student placement probability and provides personalized recommendations to improve employability.
Built using Machine Learning and deployed as an interactive Streamlit dashboard.

---

## ✨ Features

* 🔮 **Placement Prediction** — ML model trained on 500 student records (~85% accuracy)
* 📊 **Confidence Gauge** — Visual placement probability using a speedometer-style meter
* 💡 **Smart Recommendations** — Personalized suggestions based on profile weaknesses
* 📈 **EDA Dashboard** — Visual insights into placement trends
* 🧠 **Feature Importance** — Identifies key drivers like CGPA, internships, and skills
* ⚡ **Real-time Predictions** — Instant feedback via Streamlit

---

## 🛠️ Tech Stack

| Layer         | Tools                                             |
| ------------- | ------------------------------------------------- |
| Data          | Pandas, NumPy                                     |
| ML            | Scikit-learn (Logistic Regression, Random Forest) |
| Visualization | Matplotlib, Seaborn, Plotly                       |
| App           | Streamlit                                         |
| Model Storage | Joblib                                            |

---

## 📁 Project Structure

```
placement-intelligence-system/
│
├── app/
│   └── app.py                    # Streamlit application
│
├── data/
│   ├── placement_data.csv       # Dataset (500 students)
│   └── generate_data.py         # Data generation script
│
├── models/
│   ├── placement_model.pkl      # Trained ML model
│   ├── scaler.pkl               # Feature scaler
│   ├── label_encoder.pkl        # Encoder
│   ├── features.pkl             # Feature list
│   └── model_type.pkl           # Model type
│
├── notebooks/
│   ├── eda_and_ml.py            # EDA + ML pipeline
│   └── plots/                   # Generated visualizations
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/saswati-404/placement-intelligence-system.git
cd placement-intelligence-system
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
cd app
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📊 Model Performance

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | **85%** ✅ |
| Random Forest       | 82%       |

---

## 🔑 Key Insights

* 📊 CGPA plays a major role (Placed avg: **7.46**, Not placed: **6.67**)
* 💼 Internships drastically improve outcomes (**91.9% vs 39.4%**)
* 📉 Backlogs negatively impact placement probability
* 🧠 Skills score significantly influences placement outcomes

---

## 🎯 Impact

This system helps students:

* Understand their placement readiness
* Identify weak areas
* Take targeted actions to improve their chances

---

## 👤 Author

**Saswati Mishra**
B.Tech Student | Data Analytics & Machine Learning Enthusiast
