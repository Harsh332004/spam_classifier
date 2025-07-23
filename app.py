import streamlit as st 
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenization using regex (removes dependency on nltk punkt)
    text = re.findall(r'\b\w+\b', text)

    # Remove stopwords and punctuation
    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = []
    for i in y:
        text.append(ps.stem(i))

    return " ".join(text)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit UI
st.title('Spam or Ham')
txt = st.text_area("Enter your message here")
st.write(f"You wrote {len(txt)} characters.")

if st.button('Predict'):
    data = transform_text(txt)
    vector_input = tfidf.transform([data])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
