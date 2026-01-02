import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ================================
# CUSTOM CSS FUNCTION
# ================================
def apply_admin_styles():
    """Apply custom CSS for admin dashboard - called at the start of stats_page()"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Hide Streamlit branding */
        #MainMenu, footer, header {
            display: none !important;
        }
        
        /* Main app container */
        [data-testid="stApp"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        }
        
        .block-container {
            padding: 3rem 2rem !important;
            max-width: 1400px;
        }
        
        /* Header Section */
        .dashboard-header {
            margin-bottom: 3rem;
        }
        
        .dashboard-title {
            font-size: 2.75rem;
            font-weight: 700;
            color: #1a1a2e;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
        }
        
        .dashboard-subtitle {
            font-size: 1rem;
            font-weight: 400;
            color: #6b7280;
            margin-top: 0.5rem;
            letter-spacing: 0.01em;
        }
        
        /* Section Headers */
        .section-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #374151;
            margin: 2.5rem 0 1.25rem 0;
            padding-bottom: 0;
            border-bottom: none;
            letter-spacing: -0.01em;
        }
        
        /* Metric Cards - Simple & Clean */
        .metric-card {
            background: #ffffff !important;
            padding: 1.25rem 1.5rem !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.2s ease;
            border: 1px solid #e5e7eb !important;
            height: 100%;
        }
        
        .metric-card:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        }
        
        .metric-label {
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #6b7280 !important;
            margin-bottom: 0.5rem !important;
        }
        
        .metric-value {
            font-size: 1.875rem !important;
            font-weight: 600 !important;
            color: #111827 !important;
            line-height: 1.2;
            margin-bottom: 0.25rem !important;
        }
        
        .metric-change {
            font-size: 0.813rem !important;
            font-weight: 400 !important;
            color: #9ca3af !important;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        /* Chart Container */
        .chart-container {
            background: #ffffff !important;
            padding: 1.5rem !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid #e5e7eb !important;
            margin-bottom: 1.5rem !important;
        }
        
        .chart-container h4 {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #374151 !important;
            margin: 0 0 1.25rem 0 !important;
            letter-spacing: -0.01em;
        }
        
        /* Info Box */
        .info-box {
            background: #ffffff !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            border-left: 3px solid #3b82f6 !important;
            margin: 1.5rem 0 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
            font-size: 0.938rem !important;
            line-height: 1.8;
            color: #4b5563 !important;
        }
        
        .info-box strong {
            color: #1a1a2e !important;
            font-weight: 600 !important;
        }
        
        .info-box-warning {
            border-left-color: #f59e0b !important;
            background: #fffbeb !important;
        }
        
        .info-box-warning h4 {
            color: #92400e !important;
            margin: 0 0 0.5rem 0 !important;
            font-size: 1.125rem !important;
        }
        
        .info-box-warning p {
            color: #78350f !important;
            margin: 0 !important;
        }
        
        /* Streamlit Elements */
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid #f3f4f6 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
            margin-top: 1rem !important;
        }
        
        [data-testid="stExpander"] summary {
            font-weight: 500 !important;
            color: #374151 !important;
        }
        
        /* Dataframe Styling */
        [data-testid="stDataFrame"] {
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
        div[data-testid="stDataFrame"] > div {
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

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
# Load dataset statistics
# -------------------------------
def load_stats():
    """Load dataset distribution by status"""
    conn = create_connection()
    query = "SELECT status, COUNT(*) as count FROM dataset GROUP BY status"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_dataset_summary():
    """Load comprehensive dataset statistics"""
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Total records
    cursor.execute("SELECT COUNT(*) as total FROM dataset")
    total = cursor.fetchone()['total']
    
    # Records by subject
    cursor.execute("SELECT subject, COUNT(*) as count FROM dataset GROUP BY subject")
    subjects = cursor.fetchall()
    
    # Records by status
    cursor.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status")
    statuses = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return {
        'total': total,
        'subjects': pd.DataFrame(subjects),
        'statuses': pd.DataFrame(statuses)
    }

# -------------------------------
# Load training results
# -------------------------------
def load_train_results():
    """Load latest training result"""
    conn = create_connection()
    query = "SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_training_history():
    """Load all training history for trends"""
    conn = create_connection()
    query = "SELECT * FROM train_results ORDER BY timestamp DESC LIMIT 10"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# -------------------------------
# Admin Dashboard Page
# -------------------------------
def stats_page():
    # ✅ CRITICAL: Apply CSS at the start of the page
    apply_admin_styles()
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Dashboard</h1>
        <p class="dashboard-subtitle">Real-time insights and model performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        dataset_summary = load_dataset_summary()
        df_train = load_train_results()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # ================================
    # SECTION 1: Dataset Overview
    # ================================
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{dataset_summary['total']:,}</div>
            <div class="metric-change">📊 Active dataset</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        num_subjects = len(dataset_summary['subjects'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Subjects</div>
            <div class="metric-value">{num_subjects}</div>
            <div class="metric-change">📚 Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        num_statuses = len(dataset_summary['statuses'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status Types</div>
            <div class="metric-value">{num_statuses}</div>
            <div class="metric-change">🏷️ Classifications</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if not df_train.empty:
            accuracy = float(df_train['accuracy'][0]) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">{accuracy:.1f}%</div>
                <div class="metric-change">🎯 Latest model</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">N/A</div>
                <div class="metric-change">⚠️ No training data</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h4>Distribution by Status</h4>", unsafe_allow_html=True)
        
        df_status = dataset_summary['statuses']
        
        # Modern color palette - muted and sophisticated
        colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
        
        fig_status = go.Figure(data=[go.Pie(
            labels=df_status['status'],
            values=df_status['count'],
            hole=0.5,
            marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
            textfont=dict(size=14, family='Inter', color='#1a1a2e'),
            textposition='outside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_status.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#6b7280')
        )
        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h4>Distribution by Subject</h4>", unsafe_allow_html=True)
        
        df_subject = dataset_summary['subjects']
        
        fig_subject = go.Figure(data=[go.Bar(
            x=df_subject['subject'],
            y=df_subject['count'],
            marker=dict(
                color=df_subject['count'],
                colorscale=[[0, '#3b82f6'], [1, '#8b5cf6']],
                line=dict(width=0)
            ),
            text=df_subject['count'],
            textposition='outside',
            textfont=dict(size=12, family='Inter', color='#1a1a2e'),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )])
        
        fig_subject.update_layout(
            margin=dict(l=20, r=20, t=20, b=60),
            height=320,
            xaxis=dict(
                title="",
                tickfont=dict(size=11, family='Inter', color='#6b7280'),
                showgrid=False
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=11, family='Inter', color='#6b7280'),
                showgrid=True,
                gridcolor='#f3f4f6'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_subject, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed table
    with st.expander("📋 View Detailed Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**By Status:**")
            st.dataframe(df_status, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**By Subject:**")
            st.dataframe(df_subject, use_container_width=True, hide_index=True)
    
    # ================================
    # SECTION 2: Model Performance
    # ================================
    st.markdown('<div class="section-header">🎯 Model Performance</div>', unsafe_allow_html=True)
    
    if df_train.empty:
        st.markdown("""
        <div class="info-box info-box-warning">
            <h4>⚠️ No Training Results Available</h4>
            <p>No model training results found in the database yet. Train your model to see performance metrics here.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Latest training info
    st.markdown(f"""
    <div class="info-box">
        <strong>Last Trained:</strong> {df_train['timestamp'][0]}<br>
        <strong>Training ID:</strong> #{df_train.iloc[0].name + 1}
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = float(df_train['accuracy'][0])
    precision = float(df_train['prec'][0])
    recall = float(df_train['recall'][0])
    f1 = float(df_train['f1'][0])
    
    metrics_data = [
        ("Accuracy", accuracy),
        ("Precision", precision),
        ("Recall", recall),
        ("F1 Score", f1)
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.4f}</div>
                <div class="metric-change">{value*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h4>Performance Metrics</h4>", unsafe_allow_html=True)
        
        # Bar chart with modern styling
        metrics_table = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1]
        })
        
        colors_bar = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
        
        fig_bar = go.Figure(data=[go.Bar(
            x=metrics_table["Metric"],
            y=metrics_table["Value"],
            text=[f"{v:.4f}" for v in metrics_table["Value"]],
            textposition="outside",
            textfont=dict(size=12, family='Inter', color='#1a1a2e'),
            marker=dict(color=colors_bar, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>"
        )])
        
        fig_bar.update_layout(
            margin=dict(l=20, r=20, t=20, b=60),
            height=360,
            yaxis=dict(
                range=[0, 1.1],
                showgrid=True,
                gridcolor='#f3f4f6',
                tickfont=dict(size=11, family='Inter', color='#6b7280')
            ),
            xaxis=dict(
                tickfont=dict(size=11, family='Inter', color='#6b7280')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
        
        # Confusion matrix heatmap
        try:
            cm = np.array(eval(df_train["confusion_matrix"][0]))
            
            # Get class labels
            labels = ["Class 0", "Class 1"]
            if "classes" in df_train.columns and pd.notna(df_train["classes"][0]):
                try:
                    labels = list(eval(df_train["classes"][0]))
                except Exception:
                    pass
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f"Pred {l}" for l in labels],
                y=[f"Actual {l}" for l in labels],
                colorscale=[[0, '#eff6ff'], [0.5, '#93c5fd'], [1, '#3b82f6']],
                showscale=False,
                text=cm.astype(int),
                texttemplate="%{text}",
                textfont=dict(size=18, family='JetBrains Mono', color='#1a1a2e'),
                hoverongaps=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            
            fig_cm.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=360,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickfont=dict(size=11, family='Inter', color='#6b7280')),
                yaxis=dict(tickfont=dict(size=11, family='Inter', color='#6b7280'))
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying confusion matrix: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Training history trend
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("<h4>Training History Trend</h4>", unsafe_allow_html=True)
    
    df_history = load_training_history()
    if len(df_history) > 1:
        df_history = df_history.sort_values('timestamp')
        
        fig_trend = go.Figure()
        
        metrics = ['accuracy', 'prec', 'recall', 'f1']
        colors_trend = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
        names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for metric, color, name in zip(metrics, colors_trend, names):
            fig_trend.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history[metric],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3),
                marker=dict(size=8, line=dict(width=2, color='#ffffff')),
                hovertemplate=f"<b>{name}</b><br>%{{y:.4f}}<br>%{{x}}<extra></extra>"
            ))
        
        fig_trend.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=11, family='Inter', color='#6b7280')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f3f4f6',
                tickfont=dict(size=11, family='Inter', color='#6b7280')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, family='Inter', color='#6b7280')
            )
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("📊 Train the model multiple times to see performance trends over time.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance summary
    with st.expander("📊 Detailed Metrics Table"):
        detailed_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score": [accuracy, precision, recall, f1],
            "Percentage": [f"{v*100:.2f}%" for v in [accuracy, precision, recall, f1]]
        })
        st.dataframe(detailed_metrics, use_container_width=True, hide_index=True)

# ================================
# RUN PAGE
# ================================
if __name__ == "__main__":
    stats_page()