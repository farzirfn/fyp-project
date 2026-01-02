import streamlit as st
import mysql.connector
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time


# ================================
# PAGE CONFIG
# ================================


# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    .retrain-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .retrain-subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Stats cards */
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stats-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .stats-label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffc10715 0%, #fb6f9215 100%);
        border-left: 4px solid #ffc107;
    }
    
    .success-box {
        background: linear-gradient(135deg, #28a74515 0%, #20c99715 100%);
        border-left: 4px solid #28a745;
    }
    
    /* Training status */
    .training-step {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
        display: flex;
        align-items: center;
    }
    
    .step-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Dataset Class
# -------------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# DB Connection
# -------------------------------
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="fyp"
    )

# -------------------------------
# Get Dataset Info
# -------------------------------
def get_dataset_info():
    """Get current dataset statistics"""
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Total count
        cursor.execute("SELECT COUNT(*) as total FROM dataset")
        total = cursor.fetchone()['total']
        
        # Count by status
        cursor.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status")
        by_status = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {'total': total, 'by_status': by_status}
    except:
        return {'total': 0, 'by_status': []}

def get_last_training():
    """Get last training timestamp"""
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT timestamp FROM train_results ORDER BY timestamp DESC LIMIT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result['timestamp'] if result else None
    except:
        return None

# -------------------------------
# Retrain Function with Progress Tracking
# -------------------------------
def retrain_model(progress_placeholder, status_placeholder):
    start_time = time.time()
    
    try:
        # Step 1: Load dataset
        status_placeholder.markdown("""
        <div class="training-step">
            <div class="step-icon">📂</div>
            <div>Loading dataset from database...</div>
        </div>
        """, unsafe_allow_html=True)
        progress_placeholder.progress(10)
        
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT text, status FROM dataset")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=["text", "status"]).dropna()
        texts = df["text"].astype(str).tolist()
        labels = df["status"].astype(str).tolist()
        
        status_placeholder.markdown(f"""
        <div class="training-step">
            <div class="step-icon">✅</div>
            <div>Loaded {len(texts):,} records from database</div>
        </div>
        """, unsafe_allow_html=True)
        progress_placeholder.progress(20)

        # Step 2: Encode labels
        status_placeholder.markdown("""
        <div class="training-step">
            <div class="step-icon">🔢</div>
            <div>Encoding labels...</div>
        </div>
        """, unsafe_allow_html=True)
        
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        progress_placeholder.progress(25)

        # Step 3: Train-test split
        status_placeholder.markdown("""
        <div class="training-step">
            <div class="step-icon">✂️</div>
            <div>Splitting data (80% train, 20% test)...</div>
        </div>
        """, unsafe_allow_html=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_encoded, test_size=0.2, random_state=42
        )
        
        progress_placeholder.progress(30)

        # Step 4: Load tokenizer
        status_placeholder.markdown("""
        <div class="training-step">
            <div class="step-icon">🔤</div>
            <div>Loading DistilBERT tokenizer...</div>
        </div>
        """, unsafe_allow_html=True)
        
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        train_dataset = NewsDataset(X_train, y_train, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        progress_placeholder.progress(35)

        # Step 5: Load model
        status_placeholder.markdown("""
        <div class="training-step">
            <div class="step-icon">🤖</div>
            <div>Loading DistilBERT model...</div>
        </div>
        """, unsafe_allow_html=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        num_training_steps = len(train_loader) * 2
        lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        
        progress_placeholder.progress(40)

        # Step 6: Training
        model.train()
        for epoch in range(2):
            status_placeholder.markdown(f"""
            <div class="training-step">
                <div class="step-icon">🔥</div>
                <div>Training Epoch {epoch + 1}/2...</div>
            </div>
            """, unsafe_allow_html=True)
            
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                epoch_losses.append(loss.item())
                
                # Update progress
                epoch_progress = 40 + (epoch * 25) + (batch_idx / len(train_loader) * 25)
                progress_placeholder.progress(int(epoch_progress))
            
            avg_loss = np.mean(epoch_losses)
            status_placeholder.markdown(f"""
            <div class="training-step">
                <div class="step-icon">📊</div>
                <div>Epoch {epoch + 1} completed - Average Loss: {avg_loss:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        progress_placeholder.progress(90)

        # Step 7: Evaluation
        status_placeholder.markdown("""
        <div class="training-step">
            <div class="step-icon">🎯</div>
            <div>Evaluating model on test set...</div>
        </div>
        """, unsafe_allow_html=True)
        
        test_dataset = NewsDataset(X_test, y_test, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)

        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred_ids)
                true_labels.extend(batch["labels"].cpu().numpy())

        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, average="binary")
        rec = recall_score(true_labels, preds, average="binary")
        f1 = f1_score(true_labels, preds, average="binary")
        cm = confusion_matrix(true_labels, preds)

        progress_placeholder.progress(95)

        # Step 8: Save model
        status_placeholder.markdown("""
        <div class="training-step">
            <div class="step-icon">💾</div>
            <div>Saving model, tokenizer, and label encoder...</div>
        </div>
        """, unsafe_allow_html=True)
        
        model.save_pretrained("fake_news_distilbert", safe_serialization=False)
        tokenizer.save_pretrained("fake_news_distilbert")
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        # Step 9: Save metrics to DB
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO train_results (accuracy, prec, recall, f1, confusion_matrix, classes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (acc, prec, rec, f1, str(cm.tolist()), str(label_encoder.classes_.tolist())))
        conn.commit()
        cursor.close()
        conn.close()

        progress_placeholder.progress(100)
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': True,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'classes': label_encoder.classes_,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'elapsed_time': elapsed_time
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# -------------------------------
# Streamlit Page
# -------------------------------
def model_page():
    # Header
    st.markdown('<h1 class="retrain-title">🤖 Model Retraining</h1>', unsafe_allow_html=True)
    st.markdown('<p class="retrain-subtitle">Retrain the DistilBERT model with latest dataset</p>', unsafe_allow_html=True)
    
    # Get dataset info
    dataset_info = get_dataset_info()
    last_training = get_last_training()
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Total Records</div>
            <div class="stats-value">{dataset_info['total']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if dataset_info['by_status']:
            fake_count = next((item['count'] for item in dataset_info['by_status'] if 'fake' in str(item['status']).lower()), 0)
            st.markdown(f"""
            <div class="stats-card" style="border-left-color: #e45756;">
                <div class="stats-label">Fake News</div>
                <div class="stats-value" style="color: #e45756;">{fake_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-label">Fake News</div>
                <div class="stats-value">0</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if dataset_info['by_status']:
            real_count = next((item['count'] for item in dataset_info['by_status'] if 'real' in str(item['status']).lower()), 0)
            st.markdown(f"""
            <div class="stats-card" style="border-left-color: #28a745;">
                <div class="stats-label">Real News</div>
                <div class="stats-value" style="color: #28a745;">{real_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-label">Real News</div>
                <div class="stats-value">0</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        device_type = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.markdown(f"""
        <div class="stats-card" style="border-left-color: #f58518;">
            <div class="stats-label">Device</div>
            <div class="stats-value" style="font-size: 1.5rem; color: #f58518;">{device_type}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Last training info
    if last_training:
        st.markdown(f"""
        <div class="info-box">
            <strong>📅 Last Training:</strong> {last_training}<br>
            <strong>💡 Tip:</strong> Retrain the model when you have new data or want to improve accuracy.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ No Training History</strong><br>
            This model has not been trained yet. Click the button below to start training.
        </div>
        """, unsafe_allow_html=True)
    
    # Training info
    st.markdown("### ℹ️ Training Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 What happens during retraining:</h4>
            <ol>
                <li>📂 Load all data from database</li>
                <li>🔢 Encode labels (Real/Fake)</li>
                <li>✂️ Split data (80% train, 20% test)</li>
                <li>🤖 Load DistilBERT model</li>
                <li>🔥 Train for 2 epochs</li>
                <li>🎯 Evaluate on test set</li>
                <li>💾 Save model and metrics</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box warning-box">
            <h4>⚠️ Important Notes:</h4>
            <ul>
                <li>Training may take 5-15 minutes</li>
                <li>Requires sufficient memory</li>
                <li>Previous model will be overwritten</li>
                <li>Ensure dataset has enough records</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Retrain button
    st.markdown("### 🚀 Start Retraining")
    
    if dataset_info['total'] < 100:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Warning:</strong> Dataset has less than 100 records. Recommended minimum is 1000 records for good model performance.
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🤖 Start Retraining", use_container_width=True):
            if dataset_info['total'] == 0:
                st.error("❌ Cannot train: No data in database. Please upload dataset first.")
            else:
                # Create placeholders for progress
                progress_placeholder = st.progress(0)
                status_placeholder = st.empty()
                
                # Run training
                result = retrain_model(progress_placeholder, status_placeholder)
                
                # Clear progress
                progress_placeholder.empty()
                status_placeholder.empty()
                
                if result['success']:
                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Training Completed Successfully!</h3>
                        <p><strong>⏱️ Time Elapsed:</strong> {result['elapsed_time']:.1f} seconds</p>
                        <p><strong>📊 Training Set:</strong> {result['train_size']:,} samples</p>
                        <p><strong>🎯 Test Set:</strong> {result['test_size']:,} samples</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics display
                    st.markdown("### 📊 Model Performance")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    metrics_data = [
                        ("Accuracy", result['accuracy'], "🎯", "#667eea"),
                        ("Precision", result['precision'], "🎲", "#f58518"),
                        ("Recall", result['recall'], "🔍", "#e45756"),
                        ("F1 Score", result['f1'], "⚖️", "#72b7b2")
                    ]
                    
                    for col, (label, value, icon, color) in zip([col1, col2, col3, col4], metrics_data):
                        with col:
                            st.markdown(f"""
                            <div class="stats-card" style="border-left-color: {color};">
                                <div class="stats-label">{icon} {label}</div>
                                <div class="stats-value" style="color: {color};">{value:.4f}</div>
                                <div style="color: #666; font-size: 0.9rem;">{value*100:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Confusion Matrix
                    st.markdown("### 🔥 Confusion Matrix")
                    
                    cm = result['confusion_matrix']
                    classes = result['classes']
                    
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=[f"Predicted<br>{c}" for c in classes],
                        y=[f"Actual<br>{c}" for c in classes],
                        colorscale='Blues',
                        showscale=True,
                        text=cm.astype(int),
                        texttemplate="%{text}",
                        textfont={"size": 20},
                        hoverongaps=False
                    ))
                    
                    fig_cm.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=20),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    st.balloons()
                    
                else:
                    st.error(f"❌ Training failed: {result['error']}")

# ================================
# RUN PAGE
# ================================
if __name__ == "__main__":
    model_page()