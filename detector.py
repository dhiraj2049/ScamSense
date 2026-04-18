import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Simple Dataset (Replace this with your CSV later)
data = {
    'review': [
        "I love this product, it changed my life! Best ever!", 
        "Terrible quality, broke after one day. Waste of money.",
        "AMAZING PRODUCT!!! CLICK HERE FOR DISCOUNT!!!", # Fake/Spammy
        "The battery life is decent, but the screen is a bit dim.",
        "Buy now cheap price best quality luxury luxury luxury", # Fake/Spammy
        "Shipping was fast and the packaging was secure."
    ],
    'label': [0, 0, 1, 0, 1, 0] # 0 = Genuine, 1 = Fake
}

df = pd.DataFrame(data)

# 2. Text Preprocessing (TF-IDF)
# This converts words into a numerical matrix based on their importance
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['review'])
y = df['label']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Quick Test
def predict_review(text):
    vectorized_text = tfidf.transform([text])
    prediction = model.predict(vectorized_text)
    return "🚩 FAKE/SPAM" if prediction[0] == 1 else "✅ GENUINE"

# Try it out
sample = "BEST DEALS EVER!! VISIT SITE FOR 50% OFF"
print(f"Review: {sample} \nResult: {predict_review(sample)}")