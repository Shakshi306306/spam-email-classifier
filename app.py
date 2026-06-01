import streamlit as st
import joblib
import string

from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = ''.join([
        char for char in text
        if char not in string.punctuation
    ])

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return ' '.join(words)

# Streamlit UI
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Classifier")

st.write("Enter a message below to check whether it is spam or not.")

# User input
message = st.text_area(
    "Enter Email or SMS Message"
)

# Prediction button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:

        cleaned_message = clean_text(message)

        vector_input = vectorizer.transform(
            [cleaned_message]
        )

        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")
            


