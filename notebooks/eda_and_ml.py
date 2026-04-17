"""
PLACEMENT INTELLIGENCE SYSTEM
EDA + Machine Learning Pipeline
Run this file to perform analysis and train models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── SETUP ────────────────────────────────────────────────────────────────────
os.makedirs('../models', exist_ok=True)
os.makedirs('../notebooks/plots', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'placed': '#2ecc71', 'not_placed': '#e74c3c', 'accent': '#3498db', 'dark': '#2c3e50'}

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PLACEMENT INTELLIGENCE SYSTEM — EDA & ML PIPELINE")
print("=" * 60)

df = pd.read_csv('../data/placement_data.csv')
print(f"\n✅ Dataset loaded: {df.shape[0]} students, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Stats:\n{df.describe().round(2)}")
print(f"\nPlacement Rate: {df['placed'].mean()*100:.1f}%")

# ─── EDA PLOTS ────────────────────────────────────────────────────────────────
print("\n📊 Generating EDA plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Placement Intelligence System — EDA Dashboard', fontsize=16, fontweight='bold', y=1.01)

# 1. Placement Count
ax1 = axes[0, 0]
counts = df['placed'].value_counts()
bars = ax1.bar(['Not Placed', 'Placed'], counts.values,
               color=[COLORS['not_placed'], COLORS['placed']], edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')
ax1.set_title('Placement Status Distribution', fontweight='bold')
ax1.set_ylabel('Number of Students')

# 2. CGPA vs Placement
ax2 = axes[0, 1]
placed_cgpa = df[df['placed'] == 1]['cgpa']
not_placed_cgpa = df[df['placed'] == 0]['cgpa']
ax2.hist(not_placed_cgpa, bins=20, alpha=0.7, color=COLORS['not_placed'], label='Not Placed', edgecolor='white')
ax2.hist(placed_cgpa, bins=20, alpha=0.7, color=COLORS['placed'], label='Placed', edgecolor='white')
ax2.axvline(placed_cgpa.mean(), color='green', linestyle='--', linewidth=2)
ax2.axvline(not_placed_cgpa.mean(), color='red', linestyle='--', linewidth=2)
ax2.set_title('CGPA Distribution by Placement', fontweight='bold')
ax2.set_xlabel('CGPA')
ax2.legend()

# 3. Internships vs Placement Rate
ax3 = axes[0, 2]
intern_rate = df.groupby('internships')['placed'].mean() * 100
bars3 = ax3.bar(intern_rate.index, intern_rate.values, color=COLORS['accent'], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars3, intern_rate.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}%', ha='center', fontweight='bold')
ax3.set_title('Placement Rate by Internships', fontweight='bold')
ax3.set_xlabel('Number of Internships')
ax3.set_ylabel('Placement Rate (%)')
ax3.set_xticks(intern_rate.index)

# 4. Salary Distribution
ax4 = axes[1, 0]
salary_data = df[df['salary'] > 0]['salary'] / 100000
ax4.hist(salary_data, bins=25, color=COLORS['accent'], edgecolor='white', linewidth=1)
ax4.axvline(salary_data.mean(), color=COLORS['dark'], linestyle='--', linewidth=2, label=f'Mean: ₹{salary_data.mean():.1f}L')
ax4.set_title('Salary Distribution (Placed Students)', fontweight='bold')
ax4.set_xlabel('Salary (Lakhs ₹)')
ax4.legend()

# 5. Skills Score vs Placement
ax5 = axes[1, 1]
ax5.boxplot([df[df['placed']==0]['skills_score'], df[df['placed']==1]['skills_score']],
            labels=['Not Placed', 'Placed'], patch_artist=True,
            boxprops=dict(facecolor='lightblue', color=COLORS['dark']),
            medianprops=dict(color=COLORS['dark'], linewidth=2))
ax5.set_title('Skills Score by Placement', fontweight='bold')
ax5.set_ylabel('Skills Score')

# 6. Stream-wise Placement Rate
ax6 = axes[1, 2]
stream_rate = df.groupby('stream')['placed'].mean() * 100
colors_stream = [COLORS['placed'] if v > 65 else COLORS['not_placed'] for v in stream_rate.values]
bars6 = ax6.bar(stream_rate.index, stream_rate.values, color=colors_stream, edgecolor='white')
for bar, val in zip(bars6, stream_rate.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
ax6.set_title('Placement Rate by Stream', fontweight='bold')
ax6.set_ylabel('Placement Rate (%)')

plt.tight_layout()
plt.savefig('../notebooks/plots/eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ EDA dashboard saved to notebooks/plots/eda_dashboard.png")

# Correlation heatmap
fig2, ax = plt.subplots(figsize=(10, 7))
numeric_cols = ['cgpa', 'internships', 'projects', 'skills_score', 'backlogs', 'communication', 'aptitude_score', 'placed']
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0, mask=mask,
            ax=ax, cbar_kws={'label': 'Correlation'}, square=True)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('../notebooks/plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Correlation heatmap saved")

# ─── KEY INSIGHTS ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  KEY INSIGHTS FROM EDA")
print("=" * 60)
avg_cgpa_placed = df[df['placed']==1]['cgpa'].mean()
avg_cgpa_not = df[df['placed']==0]['cgpa'].mean()
print(f"\n📌 Insight 1: CGPA Impact")
print(f"   Placed students avg CGPA:     {avg_cgpa_placed:.2f}")
print(f"   Not-placed students avg CGPA: {avg_cgpa_not:.2f}")
print(f"   → CGPA difference of {avg_cgpa_placed - avg_cgpa_not:.2f} points matters significantly")

intern_0 = df[df['internships']==0]['placed'].mean()*100
intern_2 = df[df['internships']==2]['placed'].mean()*100
print(f"\n📌 Insight 2: Internship Impact")
print(f"   Placement rate with 0 internships: {intern_0:.1f}%")
print(f"   Placement rate with 2 internships: {intern_2:.1f}%")
print(f"   → Internships boost placement chances significantly")

print(f"\n📌 Insight 3: Skills Score")
print(f"   Avg skills score (placed):     {df[df['placed']==1]['skills_score'].mean():.1f}")
print(f"   Avg skills score (not placed): {df[df['placed']==0]['skills_score'].mean():.1f}")

# ─── ML PIPELINE ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MACHINE LEARNING PIPELINE")
print("=" * 60)

# Encode stream
le = LabelEncoder()
df['stream_encoded'] = le.fit_transform(df['stream'])
joblib.dump(le, '../models/label_encoder.pkl')

features = ['cgpa', 'internships', 'projects', 'skills_score', 'backlogs', 'communication', 'aptitude_score', 'stream_encoded']
X = df[features]
y = df['placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n✅ Train set: {len(X_train)} | Test set: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, '../models/scaler.pkl')

# Model 1: Logistic Regression
print("\n🤖 Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {lr_acc*100:.2f}%")
print(f"   Classification Report:\n{classification_report(y_test, lr_pred, target_names=['Not Placed','Placed'])}")

# Model 2: Random Forest
print("🤖 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   Accuracy: {rf_acc*100:.2f}%")
print(f"   Classification Report:\n{classification_report(y_test, rf_pred, target_names=['Not Placed','Placed'])}")

# Pick best model
best_model = rf_model if rf_acc >= lr_acc else lr_model
best_name = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"\n🏆 Best Model: {best_name} ({max(rf_acc, lr_acc)*100:.2f}% accuracy)")
joblib.dump(best_model, '../models/placement_model.pkl')
joblib.dump(features, '../models/features.pkl')
joblib.dump(best_name, '../models/model_type.pkl')
print("✅ Models saved to models/ folder")

# Feature Importance
print("\n📊 Generating Feature Importance plot...")
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)

fig3, ax = plt.subplots(figsize=(9, 6))
colors_fi = [COLORS['placed'] if v > feat_df['Importance'].median() else COLORS['accent'] for v in feat_df['Importance']]
bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors_fi, edgecolor='white')
for bar, val in zip(bars, feat_df['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_xlim(0, feat_df['Importance'].max() * 1.2)
plt.tight_layout()
plt.savefig('../notebooks/plots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Feature importance plot saved")

top3 = feat_df.nlargest(3, 'Importance')['Feature'].tolist()
print(f"\n🔑 Top 3 Features: {', '.join(top3)}")
print("\n✅ All done! Models and plots are ready.")
print("=" * 60)
