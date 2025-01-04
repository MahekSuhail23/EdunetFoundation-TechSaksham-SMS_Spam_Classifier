import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# CSS to set background color and beautify the app
st.markdown(
    """
    <style>
   /* Change background color for the entire Streamlit app */
    .stApp {
        background-color: #E6F7FF;  /* Apply blue background */
    }
    
    /* Optionally, you can target the content area directly */
    .block-container {
        background-color: #FFFFF0;  /* mid area */
    }

    /* Additional customizations */
    body {
        font-family: 'Arial', sans-serif;  /* Font settings */
    }

    /* Title styling */
    .css-18e3th9 {
        background-color: #4CAF50;  /* Green color for the title */
        color: white;
        padding: 10px;
        text-align: center;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    /* Styling the input box */
    .stTextInput > div > input {
        border-radius: 8px;
        border: 2px solid #4CAF50;  /* Green border */
        padding: 10px;
        font-size: 16px;
        width: 80%;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;  /* Green background */
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: blue;  /* Darker green on hover */
    }

    /* Styling the headers for results */
    h1, h2, h3 {
        color: #333;
        font-family: 'Arial', sans-serif;
    }

    /* Spam and Not Spam results */
    .css-1v3o67v {
        background-color: #f8d7da;  /* Light red for Spam */
        color: #721c24;
        padding: 10px;
        border-radius: 10px;
    }
    .css-1v3o67v span {
        font-weight: bold;
    }

    </style>
    """, unsafe_allow_html=True)

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit UI
st.title("SMS Spam Detection Model")
st.write("Welcome to the SMS Spam Detection Model!")
st.write("Intern at Edunet Foundation")
st.write("Mahek Shaikh")
# Input box for user to enter SMS
input_sms = st.text_input("Enter the SMS")

# Prediction button
if st.button('Predict'):

    # 1. preprocess the input SMS
    transformed_sms = transform_text(input_sms)
    # 2. vectorize the text using the pre-trained vectorizer
    vector_input = tk.transform([transformed_sms])
    # 3. predict the result
    result = model.predict(vector_input)[0]
    # 4. Display prediction result
    if result == 1:
        st.markdown('<div class="css-1v3o67v"><h3>Spam</h3></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="css-1v3o67v"><h3>Not Spam</h3></div>', unsafe_allow_html=True)
