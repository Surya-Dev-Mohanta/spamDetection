import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="centered")

# Custom CSS for a polished frontend
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #ced4da;
        padding: 15px;
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #007bff;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .result-box-spam {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4d4d;
        padding: 20px;
        border-radius: 8px;
        color: #b30000;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .result-box-safe {
        background-color: #e6ffe6;
        border-left: 5px solid #2eb82e;
        padding: 20px;
        border-radius: 8px;
        color: #1f7a1f;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .title-text {
        text-align: center;
        color: #343a40;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Dataset Generation (Exactly 100 Messages)
# ==========================================
@st.cache_data
def load_data():
    # Core distinct messages
    base_spam = [
        "Win a free iPhone now! Click here.", 
        "URGENT: Your bank account is locked. Verify immediately.", 
        "Claim your $1000 Walmart gift card today.", 
        "Exclusive offer just for you. Buy one get one free!", 
        "Earn $5000 a week working from home! Ask me how."
    ]
    base_safe = [
        "Hey, what time are we meeting tomorrow?", 
        "Can you send over the notes from class?", 
        "I'm picking up groceries, do you need anything?", 
        "Let's catch up later today.", 
        "The project deadline has been moved to Friday."
    ]
    
    # Multiplying and adding slight variations to create exactly 100 unique messages
    # 50 Spam Messages
    spam_data = [f"{msg} (ID: {i})" for i, msg in enumerate(base_spam * 10)]
    # 50 Safe Messages
    safe_data = [f"{msg} (Ref: {i})" for i, msg in enumerate(base_safe * 10)]
    
    # Combine into a DataFrame
    df = pd.DataFrame({
        'message': spam_data + safe_data,
        'label': ['spam'] * 50 + ['safe'] * 50
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==========================================
# 3. Model Training Pipeline
# ==========================================
@st.cache_resource
def train_model(df):
    # Pipeline: TF-IDF extracts features from text, Naive Bayes classifies them
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['message'], df['label'])
    return model

# Initialize Data and Model
data = load_data()
model = train_model(data)

# ==========================================
# 4. Frontend UI
# ==========================================
st.markdown("<h1 class='title-text'>🛡️ Smart Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>Analyze text messages in real-time to detect malicious or spam content.</p>", unsafe_allow_html=True)
st.write("---")

# User Input
user_input = st.text_area("Enter your message below:", height=150, placeholder="Type or paste a message here to check if it's spam...")

# Prediction Logic
if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Predict the label
        prediction = model.predict([user_input])[0]
        
        # Display the result with custom styled boxes
        if prediction == "spam":
            st.markdown("""
                <div class="result-box-spam">
                    🚨 WARNING: This message is classified as SPAM.
                </div>
            """, unsafe_allow_html=True)
            st.balloons() # Just a fun visual cue, you can remove if too playful
        else:
            st.markdown("""
                <div class="result-box-safe">
                    ✅ SAFE: This message looks like a normal text.
                </div>
            """, unsafe_allow_html=True)

# Footer / Debug info
st.write("---")
with st.expander("ℹ️ About the Model & Dataset"):
    st.write(f"This model was trained in memory using **{len(data)} messages** ({len(data[data['label'] == 'spam'])} spam, {len(data[data['label'] == 'safe'])} safe).")
    st.write("Algorithm: **TF-IDF Vectorizer + Multinomial Naive Bayes**")
    st.dataframe(data.head(10), use_container_width=True)
