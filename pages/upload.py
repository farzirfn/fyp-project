import streamlit as st
import pandas as pd
import mysql.connector
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict
import hashlib
import os
from contextlib import contextmanager

# ================================
# CUSTOM CSS
# ================================
def apply_upload_styles():
    """Apply custom CSS for upload page"""
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 2rem !important;
        }
        
        /* Title styling */
        .upload-title {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            margin-bottom: 0.5rem !important;
        }
        
        .upload-subtitle {
            color: #666 !important;
            font-size: 1.1rem !important;
            margin-bottom: 2rem !important;
        }
        
        /* Upload area */
        .upload-zone {
            border: 2px dashed #667eea !important;
            border-radius: 16px !important;
            padding: 3rem !important;
            text-align: center !important;
            background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%) !important;
            margin: 2rem 0 !important;
            transition: all 0.3s ease;
        }
        
        .upload-zone:hover {
            border-color: #764ba2 !important;
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%) !important;
        }
        
        /* Stats cards */
        .stats-card {
            background: white !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            text-align: center !important;
            border-left: 4px solid #667eea !important;
        }
        
        .stats-value {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #667eea !important;
            margin: 0.5rem 0 !important;
        }
        
        .stats-label {
            color: #666 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
        }
        
        .stats-change {
            color: #28a745 !important;
            font-size: 0.85rem !important;
            margin-top: 0.5rem !important;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%) !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            border-left: 4px solid #667eea !important;
            margin: 1rem 0 !important;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #ffc10715 0%, #fb6f9215 100%) !important;
            border-left: 4px solid #ffc107 !important;
        }
        
        .success-box {
            background: linear-gradient(135deg, #28a74515 0%, #20c99715 100%) !important;
            border-left: 4px solid #28a745 !important;
        }
        
        .error-box {
            background: linear-gradient(135deg, #e4575615 0%, #dc354515 100%) !important;
            border-left: 4px solid #e45756 !important;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 0.75rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ================================
# NLTK SETUP
# ================================
@st.cache_resource
def setup_nltk():
    """Download NLTK data if not already present"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    return set(stopwords.words("english")), PorterStemmer()

# Initialize NLTK resources
try:
    stop_words, stemmer = setup_nltk()
except Exception as e:
    st.error(f"❌ Error initializing NLTK: {str(e)}")
    stop_words, stemmer = set(), None

# ================================
# DATABASE CONFIGURATION
# ================================
@st.cache_resource
def get_db_config():
    """Get database configuration from environment or defaults"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'fyp')
    }

# ================================
# DATABASE CONNECTION
# ================================
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        config = get_db_config()
        conn = mysql.connector.connect(**config, autocommit=False)
        yield conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        yield None
    finally:
        if conn and conn.is_connected():
            conn.close()

def create_connection():
    """Create database connection with error handling (legacy compatibility)"""
    try:
        config = get_db_config()
        conn = mysql.connector.connect(**config, autocommit=False)
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

# ================================
# DATA VALIDATION
# ================================
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate dataframe has required columns
    Returns: (is_valid, error_message)
    """
    required_columns = ['title', 'text', 'subject', 'status']
    
    # Check if dataframe is empty
    if df.empty:
        return False, "❌ DataFrame is empty"
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"❌ Missing required columns: {', '.join(missing_cols)}"
    
    return True, "✅ Validation passed"

def validate_column_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate and clean column data
    Returns: (cleaned_df, stats_dict)
    """
    stats = {
        'empty_titles': 0,
        'empty_texts': 0,
        'invalid_status': 0
    }
    
    # Check for empty critical fields
    stats['empty_titles'] = df['title'].isna().sum()
    stats['empty_texts'] = df['text'].isna().sum()
    
    # Validate status values (should be 'real' or 'fake' or similar)
    valid_statuses = ['real', 'fake', 'true', 'false', '0', '1', 'Real', 'Fake', 'True', 'False']
    stats['invalid_status'] = (~df['status'].astype(str).isin(valid_statuses)).sum()
    
    return df, stats

# ================================
# DATA PREPROCESSING
# ================================
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove rows with missing values and clean column names"""
    original_count = len(df)
    
    # Strip whitespace from column names and lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # Remove rows with missing values in critical columns
    df = df.dropna(subset=['title', 'text', 'status'], how='any')
    
    # Fill missing subjects with 'Unknown'
    df['subject'] = df['subject'].fillna('Unknown')
    
    # Remove duplicates based on title and text
    df = df.drop_duplicates(subset=['title', 'text'], keep='first')
    
    removed_count = original_count - len(df)
    
    return df, removed_count

def clean_and_stem(text: str) -> str:
    """Clean text: lowercase, remove punctuation, remove stopwords, stem"""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and non-alphabetic chars
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    tokens = [t for t in tokens if t and t not in stop_words]
    
    # Stem each token (with error handling)
    if stemmer:
        try:
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
        except Exception:
            stemmed_tokens = tokens  # Fallback if stemming fails
    else:
        stemmed_tokens = tokens
    
    # Rejoin
    result = " ".join(stemmed_tokens)
    
    # Return original if cleaning resulted in empty string
    return result if result.strip() else text[:100]

def generate_content_hash(title: str, text: str) -> str:
    """Generate hash for duplicate detection"""
    content = f"{title}|{text}".lower().strip()
    return hashlib.md5(content.encode()).hexdigest()

# ================================
# DATABASE OPERATIONS
# ================================
def get_database_stats() -> Dict:
    """Get current database statistics"""
    try:
        with get_db_connection() as conn:
            if not conn:
                return {'total': 0, 'by_status': []}
            
            cursor = conn.cursor(dictionary=True)
            
            # Total count
            cursor.execute("SELECT COUNT(*) as total FROM dataset")
            total = cursor.fetchone()['total']
            
            # Count by status
            cursor.execute("SELECT status, COUNT(*) as count FROM dataset GROUP BY status")
            by_status = cursor.fetchall()
            
            cursor.close()
            
            return {'total': total, 'by_status': by_status}
    except Exception as e:
        st.warning(f"⚠️ Could not fetch database stats: {str(e)}")
        return {'total': 0, 'by_status': []}

def build_duplicate_check_index(conn, df: pd.DataFrame) -> set:
    """
    Build an in-memory set of existing hashes for faster duplicate checking
    Returns: set of content hashes that exist in database
    """
    try:
        cursor = conn.cursor()
        
        # Get all existing hashes from database
        cursor.execute("""
            SELECT DISTINCT MD5(CONCAT(LOWER(title), '|', LOWER(text))) as content_hash 
            FROM dataset
        """)
        
        existing_hashes = {row[0] for row in cursor.fetchall()}
        cursor.close()
        
        return existing_hashes
        
    except Exception as e:
        st.warning(f"⚠️ Could not build duplicate index: {str(e)}")
        return set()

def save_to_database(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    """
    Save dataframe to database with transaction support and optimized duplicate checking
    Returns: (success, message, stats)
    """
    conn = create_connection()
    if not conn:
        return False, "❌ Database connection failed", {}
    
    try:
        cursor = conn.cursor()
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stats = {
            'attempted': len(df),
            'inserted': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        # Build duplicate index once instead of checking each row
        status_text.text("🔄 Building duplicate check index...")
        existing_hashes = build_duplicate_check_index(conn, df)
        
        # Prepare batch insert
        status_text.text("🔄 Processing records...")
        
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            try:
                # Clean data
                title_clean = clean_and_stem(str(row.get("title", "Untitled")))
                text_clean = clean_and_stem(str(row.get("text", "")))
                subject_clean = clean_and_stem(str(row.get("subject", "Unknown")))
                status_val = str(row.get("status", "Pending")).strip()
                
                # Validate cleaned data
                if not title_clean or len(title_clean) < 3:
                    stats['errors'] += 1
                    continue
                
                if not text_clean or len(text_clean) < 10:
                    stats['errors'] += 1
                    continue
                
                # Check duplicates against in-memory set (much faster)
                content_hash = generate_content_hash(title_clean, text_clean)
                
                if content_hash in existing_hashes:
                    stats['duplicates'] += 1
                else:
                    # Insert
                    cursor.execute(
                        """INSERT INTO dataset (title, text, subject, status) 
                           VALUES (%s, %s, %s, %s)""",
                        (title_clean, text_clean, subject_clean, status_val)
                    )
                    stats['inserted'] += 1
                    
                    # Add to existing hashes to prevent duplicates within this batch
                    existing_hashes.add(content_hash)
                
            except mysql.connector.Error as e:
                stats['errors'] += 1
                if i <= 5:  # Only show first 5 errors
                    st.warning(f"⚠️ Error on row {i}: {str(e)}")
            except Exception as e:
                stats['errors'] += 1
                if i <= 5:
                    st.warning(f"⚠️ Unexpected error on row {i}: {str(e)}")
            
            # Update progress every 10 rows
            if i % 10 == 0 or i == len(df):
                progress = i / len(df)
                progress_bar.progress(progress)
                status_text.text(
                    f"🔄 Processing... {i}/{len(df)} rows "
                    f"(Inserted: {stats['inserted']}, Duplicates: {stats['duplicates']}, Errors: {stats['errors']})"
                )
        
        # Commit transaction
        conn.commit()
        
        # Get final count
        cursor.execute("SELECT COUNT(*) FROM dataset")
        final_count = cursor.fetchone()[0]
        stats['total_in_db'] = final_count
        
        cursor.close()
        conn.close()
        
        # Properly clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if stats['inserted'] > 0:
            return True, "✅ Upload completed successfully", stats
        else:
            return False, "❌ No new records inserted", stats
        
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
            conn.close()
        return False, f"❌ Database error: {str(e)}", {}
    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        return False, f"❌ Unexpected error: {str(e)}", {}

# ================================
# MAIN UPLOAD PAGE
# ================================
def upload_page():
    """Main upload page function"""
    
    # ✅ CRITICAL: Apply CSS at the start of the page
    apply_upload_styles()
    
    # Header
    st.markdown('<h1 class="upload-title">📤 Upload Dataset</h1>', unsafe_allow_html=True)
    st.markdown('<p class="upload-subtitle">Upload and preprocess your dataset with automatic cleaning</p>', unsafe_allow_html=True)
    
    # Current database stats
    db_stats = get_database_stats()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Upload Instructions</h4>
            <ul>
                <li>✅ Supported formats: CSV, XLS, XLSX</li>
                <li>✅ Required columns: <code>title</code>, <code>text</code>, <code>subject</code>, <code>status</code></li>
                <li>✅ Automatic preprocessing: lowercase, remove punctuation, remove stopwords, stemming</li>
                <li>✅ Rows with missing values will be automatically removed</li>
                <li>✅ Duplicate detection based on title and text</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-label">Current Database</div>
            <div class="stats-value">{db_stats['total']:,}</div>
            <div class="stats-change">📊 Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📁 Select File to Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, XLS, or XLSX file",
        type=["csv", "xls", "xlsx"],
        help="Upload your dataset file. Supported formats: CSV, Excel (XLS, XLSX)"
    )
    
    if uploaded_file is not None:
        try:
            # Load file based on extension
            with st.spinner("📂 Loading file..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
                elif uploaded_file.name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("❌ Unsupported file type.")
                    return
            
            st.success(f"✅ File loaded: **{uploaded_file.name}**")
            
            # Validate dataframe structure
            is_valid, validation_msg = validate_dataframe(df)
            
            if not is_valid:
                st.markdown(f"""
                <div class="error-box">
                    {validation_msg}
                    <br><br>
                    <strong>Your columns:</strong> {', '.join(df.columns.tolist())}
                    <br>
                    <strong>Required columns:</strong> title, text, subject, status
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Validate column data
            df_validated, validation_stats = validate_column_data(df)
            
            # Show validation warnings
            if any(validation_stats.values()):
                warning_messages = []
                if validation_stats['empty_titles'] > 0:
                    warning_messages.append(f"{validation_stats['empty_titles']} empty titles")
                if validation_stats['empty_texts'] > 0:
                    warning_messages.append(f"{validation_stats['empty_texts']} empty texts")
                if validation_stats['invalid_status'] > 0:
                    warning_messages.append(f"{validation_stats['invalid_status']} invalid status values")
                
                if warning_messages:
                    st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ Data quality issues detected: {', '.join(warning_messages)}
                        <br>These rows will be removed during preprocessing.
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show stats BEFORE preprocessing
            st.markdown("### 📊 Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-label">Before Preprocessing</div>
                    <div class="stats-value">{df.shape[0]:,}</div>
                    <div class="stats-change">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Preprocess data
            df_clean, removed_count = preprocess_data(df)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card" style="border-left-color: #28a745;">
                    <div class="stats-label">After Preprocessing</div>
                    <div class="stats-value" style="color: #28a745;">{df_clean.shape[0]:,}</div>
                    <div class="stats-change">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                removed_pct = (removed_count / df.shape[0] * 100) if df.shape[0] > 0 else 0
                color = "#e45756" if removed_count > 0 else "#28a745"
                st.markdown(f"""
                <div class="stats-card" style="border-left-color: {color};">
                    <div class="stats-label">Removed</div>
                    <div class="stats-value" style="color: {color};">{removed_count:,}</div>
                    <div class="stats-change">{removed_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Check if data remains after cleaning
            if len(df_clean) == 0:
                st.markdown("""
                <div class="error-box">
                    ❌ <strong>No valid data remaining after preprocessing!</strong>
                    <br>All rows were removed due to missing or invalid data.
                    <br>Please check your dataset and try again.
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Show warning if too many rows were removed
            if removed_count > 0:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>{removed_count:,} rows</strong> ({removed_pct:.1f}%) were removed due to:
                    <ul>
                        <li>Missing values in title, text, or status</li>
                        <li>Duplicate entries</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Preprocessing preview
            st.markdown("### 🔍 Preprocessing Preview")
            st.markdown("See how your data will be cleaned before saving to database:")
            
            # Create preview with before/after
            preview_size = min(5, len(df_clean))
            preview_df = df_clean.head(preview_size).copy()
            
            preview_df["title_clean"] = preview_df["title"].apply(clean_and_stem)
            preview_df["text_clean"] = preview_df["text"].apply(clean_and_stem)
            preview_df["subject_clean"] = preview_df["subject"].apply(clean_and_stem)
            
            # Show in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Original Data", "✨ Cleaned Data", "🔄 Comparison"])
            
            with tab1:
                st.dataframe(
                    df_clean[["title", "text", "subject", "status"]].head(10),
                    use_container_width=True
                )
            
            with tab2:
                cleaned_preview = pd.DataFrame({
                    "title": preview_df["title_clean"],
                    "text": preview_df["text_clean"],
                    "subject": preview_df["subject_clean"],
                    "status": preview_df["status"]
                })
                st.dataframe(cleaned_preview, use_container_width=True)
            
            with tab3:
                if len(preview_df) > 0:
                    comparison = pd.DataFrame({
                        "Column": ["Title", "Text", "Subject"],
                        "Original": [
                            str(preview_df["title"].iloc[0])[:50] + "...",
                            str(preview_df["text"].iloc[0])[:50] + "...",
                            str(preview_df["subject"].iloc[0])[:50] + "..."
                        ],
                        "Cleaned": [
                            str(preview_df["title_clean"].iloc[0])[:50] + "...",
                            str(preview_df["text_clean"].iloc[0])[:50] + "...",
                            str(preview_df["subject_clean"].iloc[0])[:50] + "..."
                        ]
                    })
                    st.dataframe(comparison, use_container_width=True, hide_index=True)
            
            # Data distribution visualization
            st.markdown("### 📊 Data Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution by status
                status_dist = df_clean['status'].value_counts()
                fig_status = px.pie(
                    values=status_dist.values,
                    names=status_dist.index,
                    title='Distribution by Status',
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )
                fig_status.update_traces(textposition='inside', textinfo='percent+label')
                fig_status.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                # Distribution by subject
                subject_dist = df_clean['subject'].value_counts().head(10)
                fig_subject = px.bar(
                    x=subject_dist.index,
                    y=subject_dist.values,
                    title='Top 10 Subjects',
                    labels={'x': 'Subject', 'y': 'Count'},
                    color=subject_dist.values,
                    color_continuous_scale='Viridis'
                )
                fig_subject.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                fig_subject.update_xaxes(tickangle=45)
                st.plotly_chart(fig_subject, use_container_width=True)
            
            # Save to database button
            st.markdown("### 💾 Save to Database")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Save to Database", use_container_width=True):
                    with st.spinner("Processing..."):
                        success, message, stats = save_to_database(df_clean)
                    
                    if success:
                        # Success message with detailed stats
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Upload Successful!</h3>
                            <p><strong>{stats['inserted']:,}</strong> new records have been processed and saved to the database.</p>
                            <hr>
                            <p><strong>📊 Upload Summary:</strong></p>
                            <ul>
                                <li>Total attempted: {stats['attempted']:,}</li>
                                <li>Successfully inserted: {stats['inserted']:,}</li>
                                <li>Duplicates skipped: {stats['duplicates']:,}</li>
                                <li>Errors: {stats['errors']:,}</li>
                                <li>Total records in database: {stats['total_in_db']:,}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                    else:
                        # Error or warning message
                        if stats:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h3>⚠️ Upload Completed with Warnings</h3>
                                <p>{message}</p>
                                <hr>
                                <p><strong>📊 Upload Summary:</strong></p>
                                <ul>
                                    <li>Total attempted: {stats.get('attempted', 0):,}</li>
                                    <li>Successfully inserted: {stats.get('inserted', 0):,}</li>
                                    <li>Duplicates skipped: {stats.get('duplicates', 0):,}</li>
                                    <li>Errors: {stats.get('errors', 0):,}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(message)
        
        except pd.errors.EmptyDataError:
            st.error("❌ The file is empty or corrupted.")
        except pd.errors.ParserError:
            st.error("❌ Error parsing file. Please check the file format.")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            # Only show full traceback in development
            if os.getenv('DEBUG', 'False').lower() == 'true':
                st.exception(e)
    
    else:
        # Show empty state
        st.markdown("""
        <div class="upload-zone">
            <h2>📁 No File Selected</h2>
            <p>Click "Browse files" above to upload your dataset</p>
            <p style="color: #999; font-size: 0.9rem;">Supported formats: CSV, XLS, XLSX</p>
            <p style="color: #999; font-size: 0.9rem;">Required columns: title, text, subject, status</p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# RUN PAGE (for standalone testing)
# ================================
if __name__ == "__main__":
    upload_page()