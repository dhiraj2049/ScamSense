import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- PAGE CONFIG ---
st.set_page_config(page_title="Review Shield Pro", page_icon="🛡️", layout="wide")

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_and_train_model():
    """
    In a professional setup, you'd load a CSV here. 
    For now, this creates a robust starting dataset.
    """
    # Expanded training data for better initial accuracy
    training_data = {
        'text': [
            "I love this product, it works perfectly and arrived on time.",
            "Great value for the money, highly recommend to everyone!",
            "The build quality is solid and the battery lasts all day.",
            "AMAZING!! CLICK HERE FOR FREE IPHONE!! BEST DEAL EVER",
            "Buy now cheap price luxury quality discount link in bio",
            "This is a scam, do not buy! They stole my credit card info.",
            "Fast shipping and the packaging was very secure.",
            "The screen is a bit dim, but otherwise a great laptop.",
            "LOSE WEIGHT FAST WITH THIS ONE WEIRD TRICK!!! CLICK NOW",
            "I've been using this for a week and it's been performing well."
        ],
        'label': [0, 0, 0, 1, 1, 1, 0, 0, 1, 0] # 0 = Genuine, 1 = Fake/Spam
    }
    df = pd.DataFrame(training_data)
    
    # Vectorizer: Converts text to numbers
    # ngram_range(1,2) helps it catch "CLICK HERE" or "BUY NOW"
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # The Model: Naive Bayes
    model = MultinomialNB()
    model.fit(X, y)
    
    return vectorizer, model

vectorizer, model = load_and_train_model()

# --- SIDEBAR ---
st.sidebar.title("Shield Settings")
st.sidebar.info("This AI detects deceptive patterns in product reviews using Natural Language Processing.")
st.sidebar.markdown("---")
st.sidebar.write("Developed by Stephan Max")

# --- MAIN UI ---
st.title("🛡️ Review Shield: Fake Review Detector")
st.markdown("Use this tool to verify if a product review is genuine or potentially deceptive.")

# Tabs for different features
tab1, tab2 = st.tabs(["Check Single Review", "Batch Analysis (CSV)"])

# --- TAB 1: SINGLE REVIEW ---
with tab1:
    st.header("Analyze a Single Review")
    user_input = st.text_area("Paste the review text here:", height=150, placeholder="Example: This product is amazing! Best thing I ever bought...")

    if st.button("Run Analysis"):
        if user_input.strip():
            # Process input
            vec_input = vectorizer.transform([user_input])
            prediction = model.predict(vec_input)
            prob = model.predict_proba(vec_input)
            
            # Show Results
            if prediction[0] == 1:
                st.error(f"### Result: 🚩 LIKELY FAKE / SPAM")
                st.write(f"Confidence Level: **{prob[0][1]:.2%}**")
            else:
                st.success(f"### Result: ✅ LIKELY GENUINE")
                st.write(f"Confidence Level: **{prob[0][0]:.2%}**")
        else:
            st.warning("Please enter some text to analyze.")

# --- TAB 2: BATCH UPLOAD ---
with tab2:
    st.header("Upload Bulk Reviews")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            # Check for a 'review' column
            # We check for common names like 'review', 'text', or 'body'
            col_name = None
            for col in ['review', 'text', 'review_body', 'content']:
                if col in batch_df.columns:
                    col_name = col
                    break
            
            if col_name:
                # Prediction
                vec_batch = vectorizer.transform(batch_df[col_name].astype(str))
                preds = model.predict(vec_batch)
                
                # Add results to dataframe
                batch_df['Analysis'] = ["🚩 FAKE" if p == 1 else "✅ GENUINE" for p in preds]
                
                # Layout for results
                col_chart, col_data = st.columns([1, 2])
                
                with col_chart:
                    st.subheader("Distribution")
                    counts = batch_df['Analysis'].value_counts()
                    st.bar_chart(counts)
                
                with col_data:
                    st.subheader("Analyzed Data")
                    st.dataframe(batch_df[[col_name, 'Analysis']], use_container_width=True)
            else:
                st.error("Error: CSV must have a column named 'review' or 'text'.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.divider()
st.caption("Disclaimer: This model is a baseline and may not catch all sophisticated fake reviews.")