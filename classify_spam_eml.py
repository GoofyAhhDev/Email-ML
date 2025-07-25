import joblib
from email import policy
from email.parser import BytesParser
import sys

def extract_email_text(eml_path):
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_content()
        for part in msg.walk():
            if part.get_content_type() == 'text/html':
                import re
                html = part.get_content()
                text = re.sub('<[^<]+?>', '', html)
                return text
    else:
        return msg.get_content()
    return ""

def main(eml_path):
    clf = joblib.load('rf_email_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    email_text = extract_email_text(eml_path)
    if not email_text:
        print("No text content found in email.")
        return

    X = vectorizer.transform([email_text])
    pred = clf.predict(X)
    print(f"Prediction for {eml_path}: {pred[0]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_eml.py path/to/email.eml")
    else:
        main(sys.argv[1])