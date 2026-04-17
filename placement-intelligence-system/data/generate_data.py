import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

cgpa = np.round(np.random.normal(7.2, 1.0, n).clip(4.0, 10.0), 2)
internships = np.random.choice([0, 1, 2, 3], n, p=[0.30, 0.40, 0.20, 0.10])
projects = np.random.choice([0, 1, 2, 3, 4], n, p=[0.10, 0.25, 0.35, 0.20, 0.10])
skills_score = np.round(np.random.normal(60, 15, n).clip(20, 100), 1)
backlogs = np.random.choice([0, 1, 2, 3], n, p=[0.60, 0.25, 0.10, 0.05])
communication = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.30, 0.30, 0.20])
aptitude_score = np.round(np.random.normal(65, 12, n).clip(30, 100), 1)
stream = np.random.choice(['CSE', 'ECE', 'ME', 'Civil', 'IT'], n, p=[0.35, 0.25, 0.15, 0.10, 0.15])

# Placement probability based on features
prob = (
    (cgpa - 4) / 6 * 0.35 +
    internships / 3 * 0.25 +
    projects / 4 * 0.15 +
    skills_score / 100 * 0.15 +
    (1 - backlogs / 3) * 0.05 +
    communication / 5 * 0.05
)
prob = prob.clip(0, 1)
noise = np.random.normal(0, 0.08, n)
prob = (prob + noise).clip(0, 1)

placed = (prob > 0.45).astype(int)

# Salary only for placed students
base_salary = 300000
salary = np.where(
    placed == 1,
    np.round((base_salary + cgpa * 40000 + internships * 30000 + skills_score * 1000 + np.random.normal(0, 50000, n)).clip(200000, 1500000), -3),
    0
)

df = pd.DataFrame({
    'cgpa': cgpa,
    'internships': internships,
    'projects': projects,
    'skills_score': skills_score,
    'backlogs': backlogs,
    'communication': communication,
    'aptitude_score': aptitude_score,
    'stream': stream,
    'placed': placed,
    'salary': salary
})

df.to_csv('placement_data.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())
print(f"\nPlacement Rate: {df['placed'].mean()*100:.1f}%")
