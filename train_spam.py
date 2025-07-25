import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

train_df = pd.read_csv('datasets/train.csv')
test_df = pd.read_csv('datasets/test.csv')

print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)

X_train_text = train_df['text']
y_train = train_df['label']
X_test_text = test_df['text']
y_test = test_df['label']

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(clf, 'rf_email_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved as rf_email_model.pkl and tfidf_vectorizer.pkl")