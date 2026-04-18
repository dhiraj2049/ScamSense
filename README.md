Project: Review Shield (Fake Product Review Detector)
Executive Summary
Review Shield is an AI-powered web application designed to identify deceptive and "spammy" product reviews. By leveraging Natural Language Processing (NLP) and Machine Learning, the tool analyzes linguistic patterns, sentiment extremity, and keyword frequencies to distinguish between genuine customer feedback and fraudulent content.

Core Features
Real-time Analysis: Users can paste any review text for an instant "Genuine vs. Fake" assessment with a confidence score.

Batch Processing: Supports CSV uploads for bulk review auditing, allowing businesses to analyze hundreds of data points at once.

Visualized Insights: Generates dynamic distribution charts to visualize the ratio of suspicious content within a dataset.

Automated Preprocessing: Utilizes TF-IDF Vectorization and N-gram analysis to catch complex spam phrases (e.g., "limited time offer," "click here").

The Tech Stack
Frontend: Streamlit (Interactive Web UI)

Language: Python 3.12

Data Science: Pandas & NumPy (Data manipulation)

Machine Learning: Scikit-learn (Multinomial Naive Bayes)

Visualization: Matplotlib & Streamlit Charts

Technical Implementation (The "How It Works")
Text Vectorization: The system transforms raw text into a numerical matrix using TF-IDF. This weights unique, descriptive words higher than common "stop words" (like the, and, or).

N-Gram Analysis: Unlike basic filters, this model looks at pairs of words (Bi-grams). This is crucial for detecting phrases like "Total scam" or "Best deal," which carry more weight than the words "Total" or "Best" individually.

Probabilistic Classification: Using the Naive Bayes algorithm, the model calculates the probability of a review being fake based on historical patterns of deceptive language.

Future Roadmap
[ ] Model Persistence: Move from on-the-fly training to pre-trained Joblib models for faster performance.

[ ] Web Scraping: Integration with BeautifulSoup/Selenium to pull reviews directly from Amazon or Yelp via URL.

[ ] Deep Learning: Implementing a BERT-based transformer for better understanding of sarcasm and complex context
