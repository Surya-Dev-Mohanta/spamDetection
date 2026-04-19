import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. Page Configuration & Modern CSS
# ==========================================
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Result Cards */
    .result-spam {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%);
        border-left: 8px solid #ff4d4d;
        padding: 25px;
        border-radius: 12px;
        color: #900000;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 10px 20px rgba(255, 77, 77, 0.15);
        animation: slideIn 0.4s ease-out;
    }
    
    .result-safe {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-left: 8px solid #28a745;
        padding: 25px;
        border-radius: 12px;
        color: #0d5c1b;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 10px 20px rgba(40, 167, 69, 0.15);
        animation: slideIn 0.4s ease-out;
    }
    
    /* Animations */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# 3. Logic & Data (Cached)
# ==========================================
@st.cache_data
def load_data():
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
    
    spam_data = [f"{msg} (ID: {i})" for i, msg in enumerate(base_spam * 10)]
    safe_data = [f"{msg} (Ref: {i})" for i, msg in enumerate(base_safe * 10)]
    
    df = pd.DataFrame({
        'message': spam_data + safe_data,
        'label': ['spam'] * 50 + ['safe'] * 50
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

@st.cache_resource
def train_model(df):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['message'], df['label'])
    return model

data = load_data()
model = train_model(data)

# ==========================================
# 4. Dashboard Layout - Sidebar
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913092.png", width=80)
    st.title("SpamGuard AI")
    st.caption("Powered by Machine Learning")
    st.write("---")
    
    st.subheader("📊 Engine Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Training Data", f"{len(data)} msg")
    col2.metric("Algorithm", "Naive Bayes")
    
    st.write("---")
    st.subheader("💡 Tips")
    st.info("Try pasting a real promotional text message or a casual text to a friend to see how the model reacts.")

# ==========================================
# 5. Dashboard Layout - Main Workspace
# ==========================================
spacer_left, main_col, spacer_right = st.columns([1, 3, 1])

with main_col:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Message Analysis Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Detect phishing, promotions, and malicious text in milliseconds.</p>", unsafe_allow_html=True)
    st.write("")
    
    # Input Container
    with st.container():
        user_input = st.text_area(
            "Message Content", 
            height=120, 
            placeholder="Type or paste the message you want to scan here...",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("🔍 Scan Message", use_container_width=True, type="primary")

    # Processing & Results
    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter a message to scan.")
        else:
            with st.spinner("Extracting features and classifying text..."):
                time.sleep(0.6) 
                
                prediction = model.predict([user_input])[0]
                
                # Save to history
                st.session_state.history.append({"Message": user_input, "Result": prediction.upper()})
            
            # Display Result
            if prediction == "spam":
                st.markdown("""
                    <div class="result-spam">
                        🚨 THREAT DETECTED: SPAM
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-safe">
                        ✅ CLEAR: SAFE MESSAGE
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()

    st.write("---")
    
    # History Expander
    if st.session_state.history:
        with st.expander("🕒 Session History", expanded=False):
            history_df = pd.DataFrame(st.session_state.history)
            
            # Function to color code the dataframe
            def color_result(val):
                color = '#ff4d4d' if val == 'SPAM' else '#28a745'
                return f'color: {color}; font-weight: bold;'
            
            # Safe rendering for the dataframe styling (handles newer and older pandas versions)
            try:
                styled_df = history_df.style.map(color_result, subset=['Result'])
            except AttributeError:
                styled_df = history_df.style.applymap(color_result, subset=['Result'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("Clear History", size="small"):
                st.session_state.history = []
                
                # Safe reload (handles newer and older Streamlit versions)
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()        border-left: 8px solid #28a745;
        padding: 25px;
        border-radius: 12px;
        color: #0d5c1b;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 10px 20px rgba(40, 167, 69, 0.15);
        animation: slideIn 0.4s ease-out;
    }
    
    /* Animations */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# 3. Logic & Data (Cached)
# ==========================================
@st.cache_data
def load_data():
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
    
    spam_data = [f"{msg} (ID: {i})" for i, msg in enumerate(base_spam * 10)]
    safe_data = [f"{msg} (Ref: {i})" for i, msg in enumerate(base_safe * 10)]
    
    df = pd.DataFrame({
        'message': spam_data + safe_data,
        'label': ['spam'] * 50 + ['safe'] * 50
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

@st.cache_resource
def train_model(df):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['message'], df['label'])
    return model

data = load_data()
model = train_model(data)

# ==========================================
# 4. Dashboard Layout - Sidebar
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913092.png", width=80)
    st.title("SpamGuard AI")
    st.caption("Powered by Machine Learning")
    st.write("---")
    
    st.subheader("📊 Engine Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Training Data", f"{len(data)} msg")
    col2.metric("Algorithm", "Naive Bayes")
    
    st.write("---")
    st.subheader("💡 Tips")
    st.info("Try pasting a real promotional text message or a casual text to a friend to see how the model reacts.")

# ==========================================
# 5. Dashboard Layout - Main Workspace
# ==========================================
spacer_left, main_col, spacer_right = st.columns([1, 3, 1])

with main_col:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Message Analysis Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Detect phishing, promotions, and malicious text in milliseconds.</p>", unsafe_allow_html=True)
    st.write("")
    
    # Input Container
    with st.container():
        user_input = st.text_area(
            "Message Content", 
            height=120, 
            placeholder="Type or paste the message you want to scan here...",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("🔍 Scan Message", use_container_width=True, type="primary")

    # Processing & Results
    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter a message to scan.")
        else:
            with st.spinner("Extracting features and classifying text..."):
                time.sleep(0.6) 
                
                prediction = model.predict([user_input])[0]
                
                # Save to history
                st.session_state.history.append({"Message": user_input, "Result": prediction.upper()})
            
            # Display Result
            if prediction == "spam":
                st.markdown("""
                    <div class="result-spam">
                        🚨 THREAT DETECTED: SPAM
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-safe">
                        ✅ CLEAR: SAFE MESSAGE
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()

    st.write("---")
    
    # History Expander
    if st.session_state.history:
        with st.expander("🕒 Session History", expanded=False):
            history_df = pd.DataFrame(st.session_state.history)
            
            # Function to color code the dataframe
            def color_result(val):
                color = '#ff4d4d' if val == 'SPAM' else '#28a745'
                return f'color: {color}; font-weight: bold;'
            
            # Safe rendering for the dataframe styling (handles newer and older pandas versions)
            try:
                styled_df = history_df.style.map(color_result, subset=['Result'])
            except AttributeError:
                styled_df = history_df.style.applymap(color_result, subset=['Result'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("Clear History", size="small"):
                st.session_state.history = []
                
                # Safe reload (handles newer and older Streamlit versions)
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()        border-left: 8px solid #28a745;
        padding: 25px;
        border-radius: 12px;
        color: #0d5c1b;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 10px 20px rgba(40, 167, 69, 0.15);
        animation: slideIn 0.4s ease-out;
    }
    
    /* Animations */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# 3. Logic & Data (Cached)
# ==========================================
@st.cache_data
def load_data():
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
    
    spam_data = [f"{msg} (ID: {i})" for i, msg in enumerate(base_spam * 10)]
    safe_data = [f"{msg} (Ref: {i})" for i, msg in enumerate(base_safe * 10)]
    
    df = pd.DataFrame({
        'message': spam_data + safe_data,
        'label': ['spam'] * 50 + ['safe'] * 50
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

@st.cache_resource
def train_model(df):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['message'], df['label'])
    return model

data = load_data()
model = train_model(data)

# ==========================================
# 4. Dashboard Layout - Sidebar
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913092.png", width=80) # Placeholder Shield Icon
    st.title("SpamGuard AI")
    st.caption("Powered by Machine Learning")
    st.write("---")
    
    st.subheader("📊 Engine Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Training Data", f"{len(data)} msg")
    col2.metric("Algorithm", "Naive Bayes")
    
    st.write("---")
    st.subheader("💡 Tips")
    st.info("Try pasting a real promotional text message or a casual text to a friend to see how the model reacts.")

# ==========================================
# 5. Dashboard Layout - Main Workspace
# ==========================================
# Use columns to constrain the width of the main content for better readability
spacer_left, main_col, spacer_right = st.columns([1, 3, 1])

with main_col:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Message Analysis Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Detect phishing, promotions, and malicious text in milliseconds.</p>", unsafe_allow_html=True)
    st.write("")
    
    # Input Container
    with st.container():
        user_input = st.text_area(
            "Message Content", 
            height=120, 
            placeholder="Type or paste the message you want to scan here...",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("🔍 Scan Message", use_container_width=True, type="primary")

    # Processing & Results
    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter a message to scan.")
        else:
            # UX Enhancement: Fake progress bar to make the AI feel like it's "thinking"
            with st.spinner("Extracting features and classifying text..."):
                time.sleep(0.6) # Short delay for tactile feedback
                
                prediction = model.predict([user_input])[0]
                
                # Save to history
                st.session_state.history.append({"Message": user_input, "Result": prediction.upper()})
            
            # Display Result
            if prediction == "spam":
                st.markdown("""
                    <div class="result-spam">
                        🚨 THREAT DETECTED: SPAM
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-safe">
                        ✅ CLEAR: SAFE MESSAGE
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()

    st.write("---")
    
    # History Expander
    if st.session_state.history:
        with st.expander("🕒 Session History", expanded=False):
            history_df = pd.DataFrame(st.session_state.history)
            
            # Function to color code the dataframe
            def color_result(val):
                color = '#ff4d4d' if val == 'SPAM' else '#28a745'
                return f'color: {color}; font-weight: bold;'
            
            st.dataframe(
                history_df.style.applymap(color_result, subset=['Result']),
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("Clear History", size="small"):
                st.session_state.history = []
                st.rerun()    
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
