import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# --------------------------------------------------
# Starter Dataset
# 0 = Genuine
# 1 = Fake / Spam
# --------------------------------------------------

def build_dataset():

    data = {
        "review": [

            "I love this product, it works perfectly and arrived on time.",
            "Very good quality and worth the price.",
            "Shipping was fast and packaging was secure.",
            "Battery life is solid and performance is smooth.",
            "The screen is decent and easy to use.",

            "AMAZING PRODUCT CLICK HERE FOR DISCOUNT NOW",
            "Buy now cheap price best quality limited offer",
            "Best product ever!!! Must buy immediately!!!",
            "Luxury deal 90 percent off click link now",
            "Free gift included hurry offer ends soon"

        ],

        "label": [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1
        ]
    }

    return pd.DataFrame(data)


# --------------------------------------------------
# Train Detector
# --------------------------------------------------

def train_detector():

    df = build_dataset()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["review"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model


# --------------------------------------------------
# Predict One Review
# --------------------------------------------------

def predict_single(text, vectorizer, model):

    X = vectorizer.transform([text])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    confidence = max(probability)

    return prediction, confidence


# --------------------------------------------------
# Predict CSV Reviews
# --------------------------------------------------

def predict_batch(df, vectorizer, model):

    possible_columns = [
        "review",
        "text",
        "content",
        "review_text"
    ]

    review_column = None

    for col in possible_columns:
        if col in df.columns:
            review_column = col
            break

    if review_column is None:
        return None

    texts = df[review_column].astype(str)

    X = vectorizer.transform(texts)

    predictions = model.predict(X)

    result = df.copy()

    result["Prediction"] = [
        "Fake / Spam" if p == 1 else "Genuine"
        for p in predictions
    ]

    return result


# --------------------------------------------------
# Local Test
# --------------------------------------------------

if __name__ == "__main__":

    vectorizer, model = train_detector()

    sample = "BEST DEAL EVER CLICK NOW LIMITED OFFER"

    label, confidence = predict_single(
        sample,
        vectorizer,
        model
    )

    result = "Fake / Spam" if label == 1 else "Genuine"

    print("Review:", sample)
    print("Prediction:", result)
    print("Confidence:", round(confidence, 2))