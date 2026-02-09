import streamlit as st
import joblib
import nltk
import string
import ssl
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="📩 SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>📩 SMS Spam Detection System</h1>
    <p style='text-align: center;'>Detect whether a message is <b>Spam</b> or <b>Ham</b></p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("model.pkl")
    vector = joblib.load("vector.pkl")
except:
    st.error("❌ model.pkl or vector.pkl missing")
    st.stop()

# ---------------- NLTK SETUP ----------------
@st.cache_resource
def load_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except:
        pass

    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']

    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res, quiet=True)

load_nltk()

# ---------------- TEXT PREPROCESSOR ----------------
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        text = str(text).lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Regex tokenizer (faster & safer)
        tokens = re.findall(r'\b\w+\b', text)

        tokens = [
            self.lemmatizer.lemmatize(word, pos='v')
            for word in tokens
            if word not in self.stop_words
        ]

        return " ".join(tokens)

preprocessor = TextPreprocessor()

# ---------------- INPUT AREA ----------------
st.subheader("✉ Enter SMS Message")

message = st.text_area(
    "Type your SMS here...",
    height=150,
    placeholder="Congratulations! You won ₹50,000..."
)

# ---------------- PREDICTION FUNCTION ----------------
def predict_sms(msg):
    clean_msg = preprocessor.preprocess(msg)
    vector_input = vector.transform([clean_msg])

    prediction = model.predict(vector_input)[0]
    prob = model.predict_proba(vector_input)[0]

    label_map = {0: "Ham", 1: "Spam"}

    return label_map[prediction], prob, clean_msg

# ---------------- PREDICTION BUTTON ----------------
if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():
        st.warning("⚠ Please enter message")
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
• TF-IDF Vectorization  
• SMOTE Balancing  
• Logistic Regression  
• NLP Preprocessing  
""")

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
🚀 Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
