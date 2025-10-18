# 🧠 Kaiburr Assessment 2025 – Task 5  
### Text Classification on Consumer Complaint Dataset  

---

## 📘 Overview
This repository contains the implementation for **Kaiburr Assessment 2025 – Task 5**, performing **multi-class text classification** on the Consumer Complaint Database.

Target categories:
| Label | Category |
|:--:|:--|
| 0 | Credit reporting, repair, or other |
| 1 | Debt collection |
| 2 | Consumer Loan |
| 3 | Mortgage |

---

## Steps Completed
1. **Exploratory Data Analysis (EDA)** – charts and word clouds saved in `/screenshots`.  
2. **Text Pre-Processing** – cleaning, stopwords removal, lemmatization.  
3. **Model Selection** – Naive Bayes, Logistic Regression, SVM, Random Forest.  
4. **Model Comparison** – accuracy comparison (bar chart).  
5. **Model Evaluation** – classification report and confusion matrix.  
6. **Prediction** – sample predictions printed to terminal.

---

## 📸 Screenshots (in repository)
- `screenshots/1_class_distribution.png` — class distribution  
- `screenshots/2_text_length_boxplot.png` — text length boxplot  
- `screenshots/wordcloud_*.png` — word clouds for each class  
- `screenshots/3_model_comparison.png` — model accuracy comparison  
- `screenshots/4_confusion_matrix.png` — confusion matrix

---

## Performance Summary
- **Best Model:** Random Forest  
- **Accuracy:** ~0.88  
- **Notes:** Consumer Loan class has few samples, so metrics for that class are unstable.

---

## How to clone & run
```bash
git clone https://github.com/YOUR_USERNAME/kaiburr-task5.git
cd kaiburr-task5

# (optional) create and activate venv
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python src/main.py
