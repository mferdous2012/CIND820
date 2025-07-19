# main.py
print("Script is running...")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

sns.set(style="whitegrid")

# -----------------------------
# Create Reports Directory
# -----------------------------
print("Creating 'reports' directory if it doesn't exist...")
os.makedirs("reports", exist_ok=True)
print("Reports directory is ready.")

# -----------------------------
# Load and Prepare Data
# -----------------------------
print("Loading and preparing data...")
fake = pd.read_csv('./data/Fake.csv')
real = pd.read_csv('./data/True.csv')
fake['label'] = 0
real['label'] = 1
df = pd.concat([fake, real], ignore_index=True)
X = df['title'].astype(str)
y = df['label']
print(f"Data loaded with {len(df)} records.")

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
print("Applying TF-IDF vectorization...")
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)
print(f"TF-IDF transformation complete. Shape: {X_tfidf.shape}")

# -----------------------------
# Cross-Validation Setup
# -----------------------------
print("Setting up 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Cross-validation configured.")

# -----------------------------
# Helper to Plot and Save Confusion Matrix
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"reports/{filename}.png")
    plt.close()

# -----------------------------
# Model Definitions and Evaluation
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

f1_results = {}

print("Starting model evaluations...")

for name, model in models.items():
    print(f"Evaluating {name}...")
    preds = cross_val_predict(model, X_tfidf, y, cv=cv)
    f1 = f1_score(y, preds)
    f1_results[name] = f1
    print(f"F1 Score: {f1:.4f}")
    
    print(f"Saving confusion matrix for {name}...")
    plot_confusion_matrix(y, preds, name, f"{name.lower().replace(' ', '_')}_cm")
    print(f"Confusion matrix saved: reports/{name.lower().replace(' ', '_')}_cm.png")

print("All models evaluated.")

# -----------------------------
# Save F1 Scores to CSV
# -----------------------------
print("Saving F1 scores to CSV...")
f1_df = pd.DataFrame(list(f1_results.items()), columns=['Model', 'F1_Score'])
f1_df.to_csv("reports/f1_scores.csv", index=False)
print("F1 scores saved to reports/f1_scores.csv")

# -----------------------------
# Save Bar Plot of F1 Scores
# -----------------------------
print("Generating F1 score comparison bar chart...")
plt.figure(figsize=(8, 5))
sns.barplot(x=f1_df['Model'], y=f1_df['F1_Score'])
plt.title("Average F1-Score by Model")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("reports/f1_score_comparison.png")
plt.close()
print("Bar chart saved to reports/f1_score_comparison.png")

print("Script completed successfully.")
