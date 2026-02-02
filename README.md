# 📩 Spam SMS Classifier using NLP

> A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing techniques.

---

## 📌 Project Overview

Spam SMS messages are a common problem that can lead to fraud, phishing attacks, and unwanted advertisements. This project builds a machine learning model that automatically detects whether an SMS message is spam or legitimate.

The project demonstrates a complete NLP workflow including text preprocessing, feature extraction using TF-IDF, model training, evaluation, and deployment using Streamlit for real-time predictions.

---

## 🧠 Features

✅ Text preprocessing and cleaning  
✅ TF-IDF vectorization for feature extraction  
✅ Multiple ML models comparison  
✅ Real-time SMS classification via Streamlit web app  
✅ Easy-to-use and beginner-friendly project structure  

---

## 📂 Project Structure

```
spam-sms-classifier/
│
├── data/                  # Dataset files
├── model
s/                # Saved trained models
├── app.py                 # Streamlit web app
├── train_model.py         # Model training script
├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── model.pkl              # Saved ML model
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**.

🔗 Dataset Source:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### Dataset Details

- 5,000+ SMS messages  
- Two categories:
  - **Ham** → Legitimate messages  
  - **Spam** → Unwanted promotional or fraudulent messages  

---

## 🔧 Text Preprocessing Steps

- Lowercasing text  
- Removing punctuation  
- Removing stopwords  
- Tokenization  
- Stemming / Lemmatization  

---

## ⚙️ Machine Learning Models Used

- 📌 Naive Bayes
- 📌 Logistic Regression

---

## 📈 Model Performance

| Model | Accuracy |
|----------|-------------|
| Naive Bayes | ~97% |
| Logistic Regression | ~95% |

*(Accuracy may vary depending on training split and preprocessing)*

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/spam-sms-classifier.git
cd spam-sms-classifier
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Model Training

```bash
python train_model.py
```

---

### Run Streamlit Web App

```bash
streamlit run app.py
```

---

## 💻 Streamlit App Features

- Enter SMS text manually  
- Instant prediction (Spam / Ham)  
- Simple and interactive UI  

---

## 📷 Demo

(Add screenshots of your Streamlit app here)

---

## 🛠 Tech Stack

- Python  
- Scikit-learn  
- Natural Language Toolkit (NLTK)  
- Pandas & NumPy  
- Streamlit  

---

## 🔮 Future Improvements

- Add Deep Learning models (LSTM / Transformers)  
- Deploy application to cloud (Streamlit Cloud / Heroku)  
- Improve UI design  
- Add multilingual spam detection  

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Vanshuu**  
GitHub: https://github.com/sumitkumar1233edeedad  

---

⭐ If you found this project useful, please consider giving it a star!
