import sys
import re
from email import policy
from email.parser import BytesParser
import joblib
import pandas as pd

STOPWORDS = set([
    'the', 'is', 'in', 'and', 'to', 'of', 'a', 'for', 'on', 'with', 'at', 'by', 'an', 'be', 'this', 'that', 'from',
    'or', 'as', 'are', 'it', 'not', 'have', 'has', 'but', 'was', 'were'
])

URGENT_KEYWORDS = [
    'urgent', 'immediately', 'asap', 'action required', 'important', 'verify', 'suspended', 'limited', 'update', 'warning'
]

feature_columns = [
    'num_words',
    'num_unique_words',
    'num_stopwords',
    'num_links',
    'num_unique_domains',
    'num_email_addresses',
    'num_spelling_errors',
    'num_urgent_keywords'
]

def extract_email_text(eml_path):
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_content()
        for part in msg.walk():
            if part.get_content_type() == 'text/html':
                html = part.get_content()
                text = re.sub('<[^<]+?>', '', html)
                return text
    else:
        return msg.get_content()
    return ""

def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words), len(set(words)), sum(1 for w in words if w.lower() in STOPWORDS)

def count_links(text):
    urls = re.findall(r'https?://[^\s]+', text)
    domains = set()
    for url in urls:
        m = re.match(r'https?://([^/]+)/?', url)
        if m:
            domains.add(m.group(1))
    return len(urls), len(domains)

def count_emails(text):
    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    return len(set(emails))

def count_spelling_errors(text):
    words = re.findall(r'\b\w+\b', text)
    spelling_errors = [w for w in words if len(w) > 15]
    return len(spelling_errors)

def count_urgent_keywords(text):
    text_lower = text.lower()
    return sum(text_lower.count(k) for k in URGENT_KEYWORDS)

def extract_features(eml_path):
    text = extract_email_text(eml_path)
    num_words, num_unique_words, num_stopwords = count_words(text)
    num_links, num_unique_domains = count_links(text)
    num_email_addresses = count_emails(text)
    num_spelling_errors = count_spelling_errors(text)
    num_urgent_keywords = count_urgent_keywords(text)
    return [
        num_words,
        num_unique_words,
        num_stopwords,
        num_links,
        num_unique_domains,
        num_email_addresses,
        num_spelling_errors,
        num_urgent_keywords
    ]

def classify_features(features_list):
    clf = joblib.load('rf_phishing_model.pkl')
    input_df = pd.DataFrame([features_list], columns=feature_columns)
    pred = clf.predict(input_df)
    label_map = {0: "LEGITIMATE", 1: "PHISHING"}
    print(f"Prediction: {label_map.get(pred[0], pred[0])}")
    return pred[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python classify_phishing_eml.py path/to/email.eml")
        sys.exit(1)
    features = extract_features(sys.argv[1])
    print("Features (use these numbers for classification):")
    print("num_words,num_unique_words,num_stopwords,num_links,num_unique_domains,num_email_addresses,num_spelling_errors,num_urgent_keywords")
    print(','.join(str(x) for x in features))
    print("\nClassifying...")
    classify_result = classify_features(features)
    classify_features(features)