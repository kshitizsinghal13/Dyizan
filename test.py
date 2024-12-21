import joblib
import nltk
import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model.predict(review_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

while True:
    user_review = input("Enter a movie review (or type 'exit' to quit): ")
    if user_review.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_review)
    print(f'The predicted sentiment is: {sentiment}')
