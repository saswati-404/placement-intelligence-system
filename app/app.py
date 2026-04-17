import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placement Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    
    .main { background-color: #f0f4f8; }
    
    .hero-box {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(26,35,126,0.3);
    }
    .hero-box h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 6px 0; }
    .hero-box p  { font-size: 1rem; opacity: 0.85; margin: 0; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-top: 4px solid #3949ab;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #1a237e; }
    .metric-card .lbl { font-size: 0.82rem; color: #555; margin-top: 4px; }

    .result-placed {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #2e7d32;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-not-placed {
        background: linear-gradient(135deg, #fce4ec, #f8bbd9);
        border: 2px solid #c62828;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-title { font-size: 1.8rem; font-weight: 700; margin: 0; }

    .rec-card {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border-left: 5px solid #f9a825;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.95rem;
    color: #212121;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #1a237e; margin: 18px 0 10px 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white; border: none; border-radius: 10px;
        padding: 14px 36px; font-size: 1rem; font-weight: 600;
        width: 100%; cursor: pointer;
        box-shadow: 0 4px 16px rgba(57,73,171,0.4);
        transition: transform 0.1s;
    }
    .stButton > button:hover { transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, '..', 'models')

@st.cache_resource
def load_models():
    model   = joblib.load(os.path.join(MODEL_DIR, 'placement_model.pkl'))
    scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le      = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    features = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
    model_type = joblib.load(os.path.join(MODEL_DIR, 'model_type.pkl'))
    return model, scaler, le, features, model_type

model, scaler, le, features, model_type = load_models()

# ─── DATA ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(BASE, '..', 'data', 'placement_data.csv')
df = pd.read_csv(DATA_PATH)

# ─── RECOMMENDATION ENGINE ────────────────────────────────────────────────────
def get_recommendations(cgpa, internships, projects, skills_score, backlogs, communication, aptitude_score, prob):
    recs = []
    if cgpa < 6.5:
        recs.append(("📚 Academic Performance", "Your CGPA is below the placement threshold. Focus on scoring above 7.0 in upcoming semesters. Attend extra classes and clear doubts regularly."))
    elif cgpa < 7.5:
        recs.append(("📚 Improve CGPA", "Your CGPA is decent but competitive companies prefer 7.5+. Aim to improve by 0.3–0.5 points."))
    
    if internships == 0:
        recs.append(("💼 Get an Internship Urgently", "You have zero internship experience — this is the #1 factor companies check. Apply on Internshala, LinkedIn, or Naukri immediately."))
    elif internships == 1:
        recs.append(("💼 Add More Internships", "One internship is good, but two or more greatly boosts your profile. Try for a part-time or remote internship."))
    
    if projects < 2:
        recs.append(("🛠️ Build More Projects", "You have fewer than 2 projects. Build at least 2–3 real-world projects and host them on GitHub. Projects speak louder than marks."))
    
    if skills_score < 55:
        recs.append(("💡 Upgrade Technical Skills", "Your skills score is low. Take free courses on Coursera, NPTEL, or YouTube. Focus on your domain: coding, design, or analytics."))
    
    if backlogs > 0:
        recs.append(("⚠️ Clear All Backlogs", f"You have {backlogs} active backlog(s). Clear them ASAP — many companies disqualify candidates with backlogs."))
    
    if communication < 3:
        recs.append(("🗣️ Communication Skills", "Low communication score. Join a public speaking club, watch English content, and practice mock interviews daily."))
    
    if aptitude_score < 55:
        recs.append(("🧮 Practice Aptitude", "Aptitude tests are part of every campus drive. Practice on IndiaBix or PrepInsta — aim for 70+ score."))
    
    if prob > 0.75 and not recs:
        recs.append(("🌟 You're on Track!", "Strong profile! Now focus on interview preparation — practice DSA, HR questions, and company-specific prep."))
    
    if not recs:
        recs.append(("✅ Good Profile", "Your profile looks solid. Keep polishing your resume and prepare well for interviews."))
    
    return recs

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Profile Input")
    st.markdown("---")
    
    cgpa         = st.slider("📊 CGPA", 4.0, 10.0, 7.5, 0.1, help="Your current CGPA out of 10")
    internships  = st.selectbox("💼 Internships Done", [0, 1, 2, 3], index=1)
    projects     = st.selectbox("🛠️ Projects Completed", [0, 1, 2, 3, 4], index=2)
    skills_score = st.slider("💡 Skills Score", 20, 100, 65, 1, help="Rate your technical skills (20–100)")
    backlogs     = st.selectbox("⚠️ Active Backlogs", [0, 1, 2, 3], index=0)
    communication = st.select_slider("🗣️ Communication (1–5)", options=[1,2,3,4,5], value=3)
    aptitude_score = st.slider("🧮 Aptitude Score", 30, 100, 65, 1)
    stream       = st.selectbox("🎓 Branch / Stream", ["CSE", "ECE", "ME", "Civil", "IT"])
    
    st.markdown("---")
    predict_btn = st.button("🔮 Predict My Placement")

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <h1>🎓 Placement Intelligence System</h1>
  <p>AI-powered placement predictor · Personalized career recommendations · Data-driven insights</p>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 EDA Dashboard", "📈 Model Info"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        stream_enc = le.transform([stream])[0]
        input_data = pd.DataFrame([{
            'cgpa': cgpa, 'internships': internships, 'projects': projects,
            'skills_score': skills_score, 'backlogs': backlogs,
            'communication': communication, 'aptitude_score': aptitude_score,
            'stream_encoded': stream_enc
        }])

        if model_type == "Logistic Regression":
             input_final = scaler.transform(input_data)
        else:
             input_final = input_data

        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]
        prob_placed = probability[1]
        
        col_r, col_m = st.columns([1, 1])
        
        with col_r:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-placed">
                  <div class="result-title">✅ LIKELY TO BE PLACED</div>
                  <div style="font-size:1.1rem; margin-top:10px; color:#2e7d32;">
                    Placement Probability: <strong>{prob_placed*100:.1f}%</strong>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-placed">
                  <div class="result-title">❌ AT RISK — NEEDS IMPROVEMENT</div>
                  <div style="font-size:1.1rem; margin-top:10px; color:#c62828;">
                    Placement Probability: <strong>{prob_placed*100:.1f}%</strong>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        if prob_placed < 0.5:
            st.warning("⚠️ Your profile needs significant improvement. Focus on internships and skills.")
        elif prob_placed < 0.75:
            st.info("⚡ You are close! Improving 1–2 areas can boost your chances.")
        else:
            st.success("🚀 Strong profile! Focus on interview preparation.")

            # Probability bar
            import plotly.graph_objects as go
            st.markdown("### 📊 Placement Confidence")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_placed * 100,
                title={'text': "Confidence (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
        
                    'bar': {
                        'color': "#2e7d32" if prob_placed > 0.7 else "#f9a825" if prob_placed > 0.5 else "#c62828"
                    },

                    'steps': [
                        {'range': [0, 50], 'color': '#ffcdd2'},   # red zone
                        {'range': [50, 75], 'color': '#fff9c4'}, # yellow zone
                        {'range': [75, 100], 'color': '#c8e6c9'} # green zone
                    ],
                }
            ))

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)
            plt.close()
        
        with col_m:
            st.markdown("<div class='section-title'>📋 Your Profile Summary</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CGPA", f"{cgpa}")
            m2.metric("Internships", f"{internships}")
            m3.metric("Projects", f"{projects}")
            m4.metric("Skills", f"{skills_score}")
        st.markdown("### 📊 Your Profile Overview")
        st.bar_chart(input_data.T)

        # Recommendations
        st.markdown("<div class='section-title'>💡 Personalized Recommendations</div>", unsafe_allow_html=True)
        recs = get_recommendations(cgpa, internships, projects, skills_score, backlogs, communication, aptitude_score, prob_placed)
        for title, text in recs:
            st.markdown(f"<div class='rec-card'><strong>{title}</strong><br>{text}</div>", unsafe_allow_html=True)
    
    else:
        st.info("👈 Fill in your profile in the **sidebar** and click **Predict My Placement** to get your result.")
        
        # Summary stats
        st.markdown("### 📊 Dataset Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='val'>{len(df)}</div><div class='lbl'>Students Analyzed</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='val'>{df['placed'].mean()*100:.0f}%</div><div class='lbl'>Placement Rate</div></div>", unsafe_allow_html=True)
        with c3:
            avg_sal = df[df['salary']>0]['salary'].mean()/100000
            st.markdown(f"<div class='metric-card'><div class='val'>₹{avg_sal:.1f}L</div><div class='lbl'>Avg Salary (Placed)</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><div class='val'>{df['cgpa'].mean():.1f}</div><div class='lbl'>Avg CGPA</div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis")
    PLOTS = os.path.join(BASE, '..', 'notebooks', 'plots')
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        eda_path = os.path.join(PLOTS, 'eda_dashboard.png')
        if os.path.exists(eda_path):
            st.image(eda_path, caption="EDA Dashboard", use_column_width=True)
    with col_e2:
        corr_path = os.path.join(PLOTS, 'correlation_heatmap.png')
        if os.path.exists(corr_path):
            st.image(corr_path, caption="Correlation Heatmap", use_column_width=True)
    
    fi_path = os.path.join(PLOTS, 'feature_importance.png')
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Feature Importance", use_column_width=True)
    
    st.markdown("### 🔑 Key Insights")
    insights = [
        ("CGPA Impact", f"Placed students have avg CGPA of **{df[df['placed']==1]['cgpa'].mean():.2f}** vs **{df[df['placed']==0]['cgpa'].mean():.2f}** for non-placed — a {df[df['placed']==1]['cgpa'].mean() - df[df['placed']==0]['cgpa'].mean():.2f} point gap."),
        ("Internship Effect", f"Students with 2+ internships have **{df[df['internships']>=2]['placed'].mean()*100:.0f}%** placement rate vs **{df[df['internships']==0]['placed'].mean()*100:.0f}%** with none."),
        ("Projects Matter", f"Students with 3+ projects show **{df[df['projects']>=3]['placed'].mean()*100:.0f}%** placement rate."),
        ("Backlogs are Costly", f"Students with backlogs have **{df[df['backlogs']>0]['placed'].mean()*100:.0f}%** placement rate vs **{df[df['backlogs']==0]['placed'].mean()*100:.0f}%** for clean record."),
    ]
    for title, text in insights:
        st.markdown(f"**📌 {title}:** {text}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Model Information")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><div class='val'>85%</div><div class='lbl'>Model Accuracy</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='val'>2</div><div class='lbl'>Models Trained</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='val'>8</div><div class='lbl'>Features Used</div></div>", unsafe_allow_html=True)
    
    st.markdown("""
    
    **Models trained:**
    - ✅ **Logistic Regression** — 85% accuracy *(selected as best)*
    - Random Forest Classifier — 82% accuracy
    
    **Features used:**
    CGPA, Internships, Projects, Skills Score, Backlogs, Communication, Aptitude Score, Stream
    
    **Tech Stack:**
    Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Streamlit · Joblib
    """)