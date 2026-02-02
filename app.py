import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="📩 SMS Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>📩 SMS Spam Detection System</h1>
    <p style='text-align: center; font-size:16px;'>
        Detect whether a message is <b>Spam</b> or <b>Ham</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("model.pkl")
    vector = joblib.load("vector.pkl")
except FileNotFoundError:
    st.error("❌ Model or Vectorizer files not found. Please ensure 'model.pkl' and 'vector.pkl' exist.")
    st.stop()

# ---------------- NLTK SETUP ----------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------- TEXT PREPROCESSOR ----------------
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(word, pos='v')
            for word in tokens
            if word not in self.stop_words
        ]
        return " ".join(tokens)

preprocessor = TextPreprocessor()

# ---------------- INPUT AREA ----------------
st.subheader("✉ Enter Your SMS Message")
message = st.text_area(
    "Type or paste your SMS here...",
    placeholder="Hey, free tickets for you! Claim now...",
    height=150
)

# ---------------- PREDICTION FUNCTION ----------------
def predict_sms(msg):
    clean_msg = preprocessor.preprocess(msg)
    vector_input = vector.transform([clean_msg])
    prediction = model.predict(vector_input)[0]
    prob = model.predict_proba(vector_input)[0]
    label_map = {0: "Ham", 1: "Spam"}
    return label_map[prediction], prob, clean_msg

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Message", use_container_width=True):
    if not message.strip():
        st.warning("⚠ Please enter a message first")
    else:
        label, prob, clean_msg = predict_sms(message)
        
        st.markdown(f"**Preprocessed Message:** `{clean_msg}`")
        st.divider()

        if label == "Spam":
            st.error("🚨 Spam Message Detected")
    
            st.progress(int(prob[1]*100))
            st.write(f"Confidence: **{round(prob[1]*100,2)}%**")
        else:
            st.success("✅ Safe Message (Ham)")
            st.balloons()
            st.progress(int(prob[0]*100))
            st.write(f"Confidence: **{round(prob[0]*100,2)}%**")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 About Project")
st.sidebar.write("""
This ML model classifies SMS messages using:
- TF-IDF Vectorization
- SMOTE Balancing
- Logistic Regression
- NLP Text Preprocessing
""")
st.sidebar.divider()
st.sidebar.write("👨‍💻 Built with Streamlit")

# ---------------- FOOTER ----------------
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class="footer">
    🚀 Built with ❤️ using Streamlit | NLP Spam Detection Project
</div>
""", unsafe_allow_html=True)