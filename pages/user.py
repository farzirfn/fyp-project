import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
import plotly.graph_objects as go
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ================================
# PAGE CONFIG (Must be first!)
# ================================

# ================================
# CUSTOM CSS FOR MODERN DESIGN
# ================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom title styling */
    .custom-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .custom-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.5s ease;
    }
    
    .real-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px solid #667eea;
    }
    
    .fake-card {
        background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%);
        border: 2px solid #f5576c;
    }
    
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-confidence {
        font-size: 1.3rem;
        color: #666;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-box {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# MODEL LOADING
# ================================
@st.cache_resource
def load_model():
    """Load the trained model, tokenizer, and label encoder"""
    model = DistilBertForSequenceClassification.from_pretrained("./fake_news_distilbert")
    tokenizer = DistilBertTokenizer.from_pretrained("./fake_news_distilbert")
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, label_encoder, device

# ================================
# PREDICTION FUNCTION
# ================================
def predict_news(model, tokenizer, label_encoder, device, text):
    """Predict whether news is real or fake"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()
    label = label_encoder.inverse_transform([pred_id])[0]

    return label, confidence, probs[0].cpu().numpy()

# ================================
# MODERN GAUGE CHART
# ================================
def create_confidence_gauge(confidence, label):
    """Create a modern gauge chart for confidence"""
    color = "#667eea" if label.lower() == "real" else "#f5576c"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 20, 'color': '#666'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ddd"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 50], 'color': '#f5f5f5'},
                {'range': [50, 75], 'color': '#e8e8e8'},
                {'range': [75, 100], 'color': '#d0d0d0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif'}
    )
    
    return fig

# ================================
# MAIN APP
# ================================
def user_home():
    # Header
    st.markdown('<h1 class="custom-title">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="custom-subtitle">Powered by AI • Analyze news articles in seconds</p>', unsafe_allow_html=True)
    
    # Spacer
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load model
    try:
        model, tokenizer, label_encoder, device = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Input section
    with st.container():
        st.markdown("### 📝 Enter News Article")
        text_input = st.text_area(
            label="Paste the news headline or article text below:",
            height=150,
            placeholder="Example: Breaking news! Scientists discover new planet...",
            label_visibility="collapsed"
        )
    
    # Predict button (centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Analyze News", use_container_width=True)
    
    # Prediction
    if predict_button:
        if text_input.strip():
            with st.spinner("🤖 Analyzing..."):
                label, confidence, probs = predict_news(model, tokenizer, label_encoder, device, text_input)
                confidence_pct = round(confidence * 100, 2)
            
            # Result section
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Result card
            if label.lower() == "real":
                st.markdown(f"""
                <div class="result-card real-card">
                    <div class="result-label">✅ REAL NEWS</div>
                    <div class="result-confidence">{confidence_pct}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card fake-card">
                    <div class="result-label">❌ FAKE NEWS</div>
                    <div class="result-confidence">{confidence_pct}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.markdown("### 📊 Confidence Analysis")
            fig = create_confidence_gauge(confidence, label)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities
            with st.expander("📈 View Detailed Probabilities"):
                df = pd.DataFrame({
                    "Class": label_encoder.classes_,
                    "Probability": [f"{p*100:.2f}%" for p in probs]
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
            
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Info section
    if not predict_button or not text_input.strip():
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4>ℹ️ How It Works</h4>
            <p>This AI-powered system uses <strong>DistilBERT</strong>, a state-of-the-art natural language processing model, to analyze news articles and determine their credibility.</p>
            <ul>
                <li>🎯 <strong>High Accuracy:</strong> Trained on thousands of verified news articles</li>
                <li>⚡ <strong>Instant Results:</strong> Get predictions in seconds</li>
                <li>🔒 <strong>Privacy First:</strong> Your text is processed locally and not stored</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #999; font-size: 0.9rem;'>"
        "⚡ Powered by DistilBERT • Made with Streamlit"
        "</p>",
        unsafe_allow_html=True
    )

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    user_home()