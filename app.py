# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from langdetect import detect
from deep_translator import GoogleTranslator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ==============================
# STREAMLIT PAGE CONFIG
# ==============================
st.set_page_config(page_title="SMS Spam Detector", page_icon="🚨", layout="centered")

# ==============================
# 2 & 3 & 4 & 5 & 6. CACHED MODEL TRAINING
# ==============================
@st.cache_resource(show_spinner="Loading data and training model... This only happens once!")
def train_model():
    # Load Data
    df = pd.read_csv("dataset_with_researched.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Preprocessing
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', 'URL', text)
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)

    df['clean_msg'] = df['message'].apply(clean_text)

    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(df['clean_msg'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Models
    nb = MultinomialNB()
    svm = SVC(probability=True)
    lr = LogisticRegression()
    
    nb.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Ensemble
    ensemble = VotingClassifier(estimators=[
        ('nb', nb),
        ('lr', lr),
        ('rf', RandomForestClassifier())
    ], voting='soft')

    ensemble.fit(X_train, y_train)
    
    return vectorizer, ensemble, stop_words

# Load the cached model
vectorizer, ensemble, stop_words = train_model()

# ==============================
# 8. SMART FEATURES
# ==============================
suspicious_words = ["win", "free", "urgent", "click", "offer", "credit", "loan", "upi", "bank"]

def clean_input(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def detect_links(text):
    return re.findall(r'(https?://\S+|bit\.ly/\S+|tinyurl\.com/\S+)', text)

def highlight_words_html(text):
    highlighted = []
    for word in text.split():
        # Remove punctuation for matching, but keep it in the final output
        clean_word = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
        if clean_word in suspicious_words:
            highlighted.append(f"<span style='color:red; font-weight:bold;'>{word}</span>")
        else:
            highlighted.append(word)
    return " ".join(highlighted)

def categorize_message(text):
    text_lower = text.lower()
    if "upi" in text_lower: return "UPI Fraud"
    elif "credit" in text_lower: return "Credit Card Scam"
    elif "loan" in text_lower: return "Loan Scam"
    elif "bank" in text_lower: return "Bank Fraud"
    else: return "General Spam"

def suggest_action(text, category, links):
    actions = []
    text_lower = text.lower()

    if links: actions.append("⚠️ Do NOT click on suspicious links.")
    if category == "Credit Card Scam":
        actions.append("💳 Never share your CVV, PIN, or OTP.")
        actions.append("🏦 Banks never ask for sensitive details via SMS.")
    elif category == "Bank Fraud":
        actions.append("🏦 Do not share OTP or login credentials.")
        actions.append("📞 Contact your bank directly if unsure.")
    elif category == "UPI Fraud":
        actions.append("📲 Do NOT approve unknown payment requests.")
        actions.append("🔐 UPI collect requests can steal money.")
    elif category == "Loan Scam":
        actions.append("💰 Avoid instant loan offers without verification.")
        actions.append("🚫 Never pay upfront processing fees.")
        
    if "win" in text_lower or "free" in text_lower:
        actions.append("🎁 If it sounds too good to be true, it's likely fake.")
    if "otp" in text_lower or "urgent" in text_lower:
        actions.append("⏳ Scammers create urgency—stay calm and verify.")
    if "call" in text_lower or "contact" in text_lower:
        actions.append("📞 Avoid calling unknown numbers from SMS.")
    if not actions:
        actions.append("⚠️ Be cautious. Verify sender before taking action.")
        
    return actions

# ==============================
# TRANSLATION FUNCTIONS
# ==============================
def translate_sms(text, target_lang):
    try:
        lang_map = {"en": "en", "hi": "hi", "or": "or"}
        if target_lang not in lang_map: return text, False
        translated = GoogleTranslator(source='auto', target=lang_map[target_lang]).translate(text)
        return translated, True
    except Exception as e:
        return text, False

def translate_actions(actions, lang_choice, translation_ok):
    if not translation_ok or lang_choice not in ["hi", "or"]: return actions
    translated_actions = []
    for act in actions:
        try:
            translated = GoogleTranslator(source='auto', target=lang_choice).translate(act)
            translated_actions.append(translated)
        except:
            translated_actions.append(act)
    return translated_actions

# ==============================
# STREAMLIT UI (FRONTEND)
# ==============================
st.title("🚨 Smart SMS Fraud Detector")
st.write("Paste an SMS message below to analyze it for spam, phishing, and financial fraud.")

# Input Form
with st.form("sms_form"):
    msg = st.text_area("Enter SMS Message:", height=150)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        lang_choice = st.selectbox("Translation:", ["Skip", "English (en)", "Hindi (hi)", "Odia (or)"])
    
    submit_button = st.form_submit_button("Analyze Message", type="primary")

# Processing and Output
if submit_button:
    if not msg.strip():
        st.warning("Please enter a message to analyze.")
    else:
        original_msg = msg
        target_lang_code = lang_choice.split("(")[-1].replace(")", "") if lang_choice != "Skip" else "skip"
        translation_ok = False
        
        # Translation Step
        with st.spinner("Analyzing message..."):
            if target_lang_code != "skip":
                msg, translation_ok = translate_sms(original_msg, target_lang_code)
                st.info(f"**Translated Message:**\n\n{msg}")

            # Prediction Step
            cleaned = clean_input(msg)
            vector = vectorizer.transform([cleaned])
            pred = ensemble.predict(vector)[0]
            prob = ensemble.predict_proba(vector)[0][1]

            st.divider()

            # Results Display
            if pred == 1:
                st.error("🚨 **SPAM / FRAUD DETECTED** 🚨")
                
                # Metrics Row
                m1, m2 = st.columns(2)
                m1.metric("Spam Confidence Score", f"{round(prob*100, 2)}%")
                m2.metric("Threat Category", categorize_message(msg))
                
                # Highlighted Text
                st.subheader("📝 Message Analysis")
                highlighted_html = highlight_words_html(original_msg)
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; color: black;'>{highlighted_html}</div>", unsafe_allow_html=True)
                
                # Links
                links = detect_links(original_msg)
                if links:
                    st.warning(f"**⚠️ Suspicious Links Found:** {', '.join(links)}")
                
                # Suggested Actions
                st.subheader("🛡️ Suggested Actions")
                actions = suggest_action(msg, categorize_message(msg), links)
                actions = translate_actions(actions, target_lang_code, translation_ok)
                
                for act in actions:
                    st.markdown(f"- {act}")
                    
            else:
                st.success("✅ **SAFE MESSAGE**")
                st.balloons()
                st.write("This message does not appear to contain spam or fraud patterns.")
                st.metric("Spam Confidence Score", f"{round(prob*100, 2)}%")
