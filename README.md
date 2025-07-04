# 📰 Fake News Detection Using News Headlines

This project is part of the CIND820 course at Toronto Metropolitan University. It focuses on detecting fake news by analyzing the textual features of news headlines using Natural Language Processing (NLP) and Machine Learning (ML) techniques.

---

## 📚 Project Overview

The goal of this project is to determine whether a given news headline is **real** or **fake**. The study investigates the effectiveness of various machine learning classifiers and textual features, including sentiment polarity, in identifying fake news content based solely on headline-level data.

---

## ✅ Research Questions

1. Can we accurately classify news headlines as fake or real using NLP techniques?
2. What words or phrases are most indicative of fake news?
3. Does sentiment polarity vary significantly between real and fake news?

---

## 📁 Repository Structure

```
fake-news-detection-headlines/
├── data/                        # Raw data files (Fake.csv, True.csv)
├── notebooks/                  # Jupyter Notebooks by stage
│   ├── EDA.ipynb
│   └── man.ipynb
├── reports/                    # Visuals and exported reports
│   ├── report.pdf or .html
│   └── visuals/
├── README.md                   # Project description and instructions
├── requirements.txt            # List of dependencies
```

---

## 📊 Models Considered

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

Each model is trained and evaluated using **5-fold cross-validation**. The evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix (for the best model)

---

## 🧪 Dataset

- **Source**: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Language**: English
- **Fields**: `title`, `text`, `subject`, `date`, `label`

Note: The dataset contains approximately 44,000 headlines labeled as either **Fake** or **Real**.

---

## 🔧 Setup Instructions

To run the notebooks:

1. Clone the repository  
   ```
   git clone https://github.com/mferdous2012/CIND820.git
   cd CIND820
   ```

2. Install dependencies  
   ```
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook  
   ```
   jupyter notebook
   ```

---

## 🎥 Video Walkthrough

A 5-minute walkthrough video presenting the code and initial results is available here:  
👉 [Watch the Video](https://your-link-to-video.com)

---

## ✍️ Authors and Acknowledgments

- **Author**: Md Mahmud Ferdous  
- **Supervisor**: Ceni Babaoglu  
- **Course**: CIND820 DAH - Big Data Analytics Project  
- **Institution**: Toronto Metropolitan University

---

## 📌 License

This project is for educational use as part of CIND820 and is not intended for commercial distribution.
