import pandas as pd
import re

STOPWORDS = set([
    'the', 'is', 'in', 'and', 'to', 'of', 'a', 'for', 'on', 'with', 'at', 'by', 'an', 'be', 'this', 'that', 'from',
    'or', 'as', 'are', 'it', 'not', 'have', 'has', 'but', 'was', 'were', 'zijn', 'niet', 'hebben', 'heeft', 'maar',
    'was', 'waren', 'de', 'het', 'een', 'en', 'voor', 'op', 'met', 'bij', 'aan', 'tot', 'uit', 'over', 'door', 'naar',
    'om', 'als', 'dan', 'zo', 'ook', 'nu', 'geen', 'welke', 'wie', 'wat', 'waarom', 'hoe'
])

URGENT_KEYWORDS = [
    'urgent', 'immediately', 'asap', 'action required', 'important', 'verify', 'suspended', 'limited', 'update', 'warning',
    'belangrijk', 'onmiddellijk', 'actie vereist', 'verifiëren', 'opgeschort', 'beperkt', 'bijwerken', 'waarschuwing'
]

EXTORTION_KEYWORDS = [
    "bitcoin", "btc", "wallet", "crypto", "cryptocurrency", "transaction", "transfer", "payment", "pay", "send money",
    "blackmail", "extortion", "ransom", "ransomware", "compromised", "hacked", "webcam", "video", "recording", "publish",
    "leak", "deadline", "hours", "48 hours", "24 hours", "immediately", "within 24 hours", "personal data", "private data",
    "spread", "share", "privacy", "threat", "threaten", "expose", "embarrass", "humiliate", "masturbate", "masturbation",
    "sexual", "intimate", "nude", "porn", "pornographic", "adult sites", "visit porn", "watch porn", "caught on camera",
    "pay in bitcoin", "send bitcoin", "bitcoin address", "btc address", "btc wallet", "crypto wallet",
    
    "bitcoin", "btc", "wallet", "cryptovaluta", "transactie", "overschrijving", "betaling", "betaal", "geld overmaken",
    "afpersing", "dreigen", "dreiging", "chantage", "losgeld", "ransom", "gecompromitteerd", "gehackt", "webcam", "video",
    "opname", "publiceren", "lekken", "deadline", "uren", "48 uur", "24 uur", "onmiddellijk", "binnen 24 uur", "persoonlijke gegevens",
    "privégegevens", "verspreiden", "delen", "privacy", "bedreigen", "blootstellen", "in verlegenheid brengen", "vernederen",
    "masturberen", "masturbatie", "seksueel", "intiem", "naakt", "porno", "pornografisch", "volwassen sites", "porno bezoeken",
    "porno kijken", "betrapt op camera", "betaal in bitcoin", "stuur bitcoin", "bitcoin adres", "btc adres", "btc wallet", "crypto wallet"
]

FEATURE_COLUMNS = [
    'num_words', 'num_unique_words', 'num_stopwords', 'num_links',
    'num_unique_domains', 'num_email_addresses', 'num_spelling_errors',
    'num_urgent_keywords', 'num_extortion_keywords', 'label'
]

def count_keywords(text, keywords):
    text_lower = str(text).lower()
    return sum(text_lower.count(k) for k in keywords)

def extract_features(text):
    words = re.findall(r'\b\w+\b', str(text))
    num_words = len(words)
    num_unique_words = len(set(words))
    num_stopwords = sum(1 for w in words if w.lower() in STOPWORDS)
    num_links = len(re.findall(r'https?://[^\s]+', str(text)))
    num_unique_domains = len(set(re.findall(r'https?://([^/]+)/?', str(text))))
    num_email_addresses = len(set(re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', str(text))))
    num_spelling_errors = sum(1 for w in words if len(w) > 15)
    num_urgent_keywords = count_keywords(text, URGENT_KEYWORDS)
    num_extortion_keywords = count_keywords(text, EXTORTION_KEYWORDS)
    return [
        num_words,
        num_unique_words,
        num_stopwords,
        num_links,
        num_unique_domains,
        num_email_addresses,
        num_spelling_errors,
        num_urgent_keywords,
        num_extortion_keywords
    ]

df_features = pd.read_csv("email_phishing_data.csv")

df_text = pd.read_csv("fraud_email_.csv")
rows = []
for idx, row in df_text.iterrows():
    features = extract_features(row['Text'])
    label = row['Class']
    rows.append(features + [label])
df_text_features = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

df_all = pd.concat([df_features, df_text_features], ignore_index=True)
df_all.to_csv("all_combined_features.csv", index=False)
print("Opgeslagen als all_combined_features.csv")