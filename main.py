# ----------------------------------------------------------
# Kaiburr Assessment 2025 - Task 5
# Text Classification on Consumer Complaint Dataset
# ----------------------------------------------------------

# ========== 1. EXPLORATORY DATA ANALYSIS & FEATURE ENGINEERING ==========
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# make sure screenshots folder exists
os.makedirs('screenshots', exist_ok=True)

print("🔹 Loading filtered dataset...")
df = pd.read_csv('data/filtered_complaints.csv')

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nCategory distribution:")
print(df['Product'].value_counts())

# ----- EDA: Class distribution -----
plt.figure(figsize=(8,4))
sns.countplot(x='Product', data=df, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Class Distribution of Complaints')
plt.tight_layout()
plt.savefig('screenshots/1_class_distribution.png')
plt.close()

# ----- EDA: Complaint length -----
df['text_length'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8,4))
sns.boxplot(x='Product', y='text_length', data=df, palette='Set2')
plt.xticks(rotation=45, ha='right')
plt.title('Complaint Text Length per Category')
plt.tight_layout()
plt.savefig('screenshots/2_text_length_boxplot.png')
plt.close()

# ----- EDA: Word clouds -----
for prod in df['Product'].unique():
    text = ' '.join(df[df['Product'] == prod]['Consumer complaint narrative'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {prod}')
    plt.tight_layout()
    plt.savefig(f'screenshots/wordcloud_{prod.replace(" ", "_")}.png')
    plt.close()

print("✅ EDA graphs generated and saved to screenshots folder.")

# ========== 2. TEXT PRE-PROCESSING ==========
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("\n🔹 Cleaning text...")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)
print("✅ Text preprocessing completed!")

# ========== 3. SELECTION OF MULTI CLASSIFICATION MODEL ==========
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

print("\n🔹 Creating TF-IDF features ...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Train/Test split done!")

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=300),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier()
}

# ========== 4. COMPARISON OF MODEL PERFORMANCE ==========
from sklearn.metrics import accuracy_score

results = {}
print("\n🔹 Training and comparing models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.3f}")

# Plot model accuracy comparison
plt.figure(figsize=(7,4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='crest')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('screenshots/3_model_comparison.png')
plt.close()

# ========== 5. MODEL EVALUATION ==========
from sklearn.metrics import classification_report, confusion_matrix
import joblib

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

print("\n✅ Best Model:", best_model_name)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix visualization
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('screenshots/4_confusion_matrix.png')
plt.close()

# Save best model and vectorizer
joblib.dump(best_model, 'src/best_model.pkl')
joblib.dump(vectorizer, 'src/vectorizer.pkl')
print(f"✅ Saved best model ({best_model_name}) and vectorizer.")

# ========== 6. PREDICTION ==========
reverse_map = {
    0: 'Credit reporting, credit repair services, or other personal consumer reports',
    1: 'Debt collection',
    2: 'Consumer Loan',
    3: 'Mortgage'
}

sample_texts = [
    "I was wrongly charged for a loan I never took.",
    "The mortgage company did not process my payment correctly.",
    "The debt collector kept calling me at work.",
    "Incorrect information is listed on my credit report."
]

print("\n🔹 Making sample predictions:")
for text in sample_texts:
    vec = vectorizer.transform([text])
    pred = best_model.predict(vec)[0]
    print(f"\nComplaint: {text}")
    print(f"Predicted Category: {reverse_map[pred]}")

print("\n🎯 Task 5 completed successfully! All EDA visuals saved to /screenshots.")
