import streamlit as st
import mysql.connector
import pandas as pd
import torch
import nlpaug.augmenter.word as naw
from transformers import DistilBertTokenizer, DistilBertModel
from scipy.stats import ks_2samp
import numpy as np
import pickle
import os
from datetime import datetime

# ================================
# SESSION STATE INITIALIZATION
# ================================
def init_session_state():
    """Initialize all session state variables to prevent data loss"""
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = None
    if 'imbalance_info' not in st.session_state:
        st.session_state['imbalance_info'] = None
    if 'minority_validation' not in st.session_state:
        st.session_state['minority_validation'] = None
    if 'df_combined' not in st.session_state:
        st.session_state['df_combined'] = None
    if 'validation_stats' not in st.session_state:
        st.session_state['validation_stats'] = None
    if 'augmentation_results' not in st.session_state:
        st.session_state['augmentation_results'] = None
    if 'progress_state' not in st.session_state:
        st.session_state['progress_state'] = {
            'current_step': 0,
            'total_steps': 0,
            'completed': False
        }

# ================================
# AUTO-SAVE TO DISK (Extra Protection)
# ================================
CACHE_DIR = "./streamlit_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def save_to_cache(key, data):
    """Save data to disk cache"""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"⚠️ Could not save cache: {e}")

def load_from_cache(key):
    """Load data from disk cache"""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load cache: {e}")
    return None

def clear_cache():
    """Clear all cache files"""
    try:
        for file in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, file))
        st.success("✅ Cache cleared!")
    except Exception as e:
        st.error(f"❌ Error clearing cache: {e}")

# -------------------------------
# DB connection
# -------------------------------
def create_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="fyp"
    )

# -------------------------------
# Load dataset from database
# -------------------------------
def load_dataset():
    """Load original dataset from database"""
    conn = create_connection()
    df = pd.read_sql("SELECT title, text, subject, status FROM dataset", conn)
    conn.close()
    return df

# -------------------------------
# Identify imbalanced classes
# -------------------------------
def identify_imbalanced_classes(df, column='subject', threshold_ratio=1.5):
    """
    Identify minority classes that need augmentation.
    
    Args:
        df: DataFrame with the dataset
        column: Column to check for imbalance ('subject' or 'status')
        threshold_ratio: If max_count/class_count > threshold_ratio, class is minority
    
    Returns:
        dict with class distribution and list of minority classes
    """
    class_counts = df[column].value_counts()
    max_count = class_counts.max()
    
    minority_classes = []
    class_info = {}
    
    for cls, count in class_counts.items():
        ratio = max_count / count
        is_minority = ratio > threshold_ratio
        
        class_info[cls] = {
            'count': count,
            'ratio': ratio,
            'is_minority': is_minority,
            'needs_augmentation': count
        }
        
        if is_minority:
            minority_classes.append(cls)
    
    return {
        'class_counts': class_counts,
        'class_info': class_info,
        'minority_classes': minority_classes,
        'max_count': max_count,
        'column': column
    }

# -------------------------------
# Augmentation with DistilBERT (nlpaug)
# -------------------------------
def augment_text_with_bert(text, num_aug=2):
    """Augment text using DistilBERT contextual word embeddings (faster than BERT)"""
    distilbert_aug = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased',
        action="substitute"
    )
    augmented = distilbert_aug.augment(text, n=num_aug)
    # Ensure it returns a list
    if isinstance(augmented, str):
        return [augmented]
    return augmented

# -------------------------------
# DistilBERT embedding
# -------------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_embedding(text):
    """Get DistilBERT embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:,0,:].squeeze().numpy()

# -------------------------------
# K-S validation (FIXED)
# -------------------------------
def ks_validate(original_text, augmented_text, threshold=0.05):
    """
    Validate augmented text using Kolmogorov-Smirnov test.
    Compares the distribution of embedding dimensions.
    Returns True if augmented text is similar enough (p > threshold).
    """
    emb_orig = get_embedding(original_text)
    emb_aug = get_embedding(augmented_text)
    
    # Ensure embeddings are 1D arrays
    if len(emb_orig.shape) > 1:
        emb_orig = emb_orig.flatten()
    if len(emb_aug.shape) > 1:
        emb_aug = emb_aug.flatten()
    
    # Perform K-S test on the full embedding vectors
    stat, p_value = ks_2samp(emb_orig, emb_aug)
    
    # Valid if p > threshold (distributions are similar)
    is_valid = p_value > threshold
    
    return is_valid, {"ks_stat": stat, "p_value": p_value}

# -------------------------------
# Validate minority class data quality (BEFORE augmentation)
# -------------------------------
def validate_minority_data_quality(df_minority):
    """
    Validate minority class data by checking internal consistency.
    Uses K-S test to compare first half vs second half of minority data.
    This ensures minority data quality is good before augmentation.
    
    Args:
        df_minority: DataFrame with minority class records
    
    Returns:
        dict with validation results
    """
    st.info(f"🔍 Validating ALL {len(df_minority)} minority class records...")
    
    minority_embeddings = []
    
    # Generate embeddings for ALL minority data
    progress_bar = st.progress(0)
    for idx, row in df_minority.iterrows():
        try:
            emb = get_embedding(row['text'])
            minority_embeddings.append(emb.flatten())
        except Exception as e:
            st.warning(f"Error processing row {idx}: {e}")
        
        # Update progress
        progress = (len(minority_embeddings)) / len(df_minority)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    
    if len(minority_embeddings) < 2:
        return {
            'is_valid': False,
            'error': 'Not enough embeddings generated',
            'total_samples': len(minority_embeddings)
        }
    
    # Split minority data into two halves for comparison
    # This checks if minority data is internally consistent
    mid_point = len(minority_embeddings) // 2
    first_half = minority_embeddings[:mid_point]
    second_half = minority_embeddings[mid_point:]
    
    # Flatten all embeddings into single distributions
    first_half_dist = np.concatenate(first_half)
    second_half_dist = np.concatenate(second_half)
    
    # Perform K-S test between two halves
    stat, p_value = ks_2samp(first_half_dist, second_half_dist)
    
    # Interpretation:
    # High p-value (> 0.05): Two halves are similar (GOOD - data is consistent)
    # Low p-value (< 0.05): Two halves differ (WARNING - data might be inconsistent)
    
    threshold = 0.05
    is_valid = p_value > threshold
    
    return {
        'is_valid': is_valid,
        'ks_stat': stat,
        'p_value': p_value,
        'total_samples': len(minority_embeddings),
        'first_half_samples': len(first_half),
        'second_half_samples': len(second_half),
        'interpretation': 'Minority data is internally consistent' if is_valid else 'Minority data shows internal inconsistency - may have quality issues'
    }

# -------------------------------
# Save combined dataset to database
# -------------------------------
def save_combined_dataset(df_combined):
    """Save combined dataset (original + valid augmented) to database"""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS combined_dataset (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            text TEXT,
            subject VARCHAR(100),
            status VARCHAR(50),
            is_augmented BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Clear existing data
    cursor.execute("TRUNCATE TABLE combined_dataset")

    # Insert combined data
    for _, row in df_combined.iterrows():
        cursor.execute("""
            INSERT INTO combined_dataset (title, text, subject, status, is_augmented)
            VALUES (%s, %s, %s, %s, %s)
        """, (row["title"], row["text"], row["subject"], row["status"], row["is_augmented"]))

    conn.commit()
    cursor.close()
    conn.close()

# -------------------------------
# Save validation statistics
# -------------------------------
def save_validation_stats(validation_stats):
    """Save K-S validation statistics to database"""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            total_original INT,
            total_augmented_generated INT,
            total_valid_augmented INT,
            total_invalid_augmented INT,
            avg_ks_stat FLOAT,
            avg_p_value FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO validation_results 
        (total_original, total_augmented_generated, total_valid_augmented, 
         total_invalid_augmented, avg_ks_stat, avg_p_value)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        validation_stats['total_original'],
        validation_stats['total_augmented_generated'],
        validation_stats['total_valid_augmented'],
        validation_stats['total_invalid_augmented'],
        validation_stats['avg_ks_stat'],
        validation_stats['avg_p_value']
    ))

    conn.commit()
    cursor.close()
    conn.close()

# -------------------------------
# Save pre-augmentation minority validation results
# -------------------------------
def save_minority_validation_to_db(validation_result, minority_classes, balance_column):
    """
    Save minority class validation results (BEFORE augmentation) to database
    
    Args:
        validation_result: dict from validate_minority_data_quality()
        minority_classes: list of minority class names
        balance_column: column name used for balancing
    """
    conn = create_connection()
    cursor = conn.cursor()

    # Create table for pre-augmentation validation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pre_augmentation_validation (
            id INT AUTO_INCREMENT PRIMARY KEY,
            balance_column VARCHAR(50),
            minority_classes TEXT,
            total_samples INT,
            first_half_samples INT,
            second_half_samples INT,
            ks_statistic FLOAT,
            p_value FLOAT,
            is_valid BOOLEAN,
            interpretation TEXT,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Convert numpy types to Python types
    is_valid_bool = bool(validation_result['is_valid'])
    ks_stat_float = float(validation_result['ks_stat'])
    p_value_float = float(validation_result['p_value'])
    total_samples_int = int(validation_result['total_samples'])
    first_half_int = int(validation_result['first_half_samples'])
    second_half_int = int(validation_result['second_half_samples'])

    # Insert validation result
    cursor.execute("""
        INSERT INTO pre_augmentation_validation 
        (balance_column, minority_classes, total_samples, first_half_samples, 
         second_half_samples, ks_statistic, p_value, is_valid, interpretation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        balance_column,
        ', '.join(map(str, minority_classes)),
        total_samples_int,
        first_half_int,
        second_half_int,
        ks_stat_float,
        p_value_float,
        is_valid_bool,
        validation_result['interpretation']
    ))

    conn.commit()
    cursor.close()
    conn.close()

# -------------------------------
# Streamlit UI with Session State
# -------------------------------
def augment_page():
    # Initialize session state
    init_session_state()
    
    st.title("📝 Text Augmentation for Imbalanced Data (DistilBERT)")
    st.markdown("**Workflow:** Load → Identify Imbalance → Augment Minority Classes → Validate → Combine → Save")
    st.info("⚡ Using DistilBERT for both augmentation and validation (30-40% faster than BERT!)")
    
    # Recovery info
    with st.expander("💾 Session Recovery Info"):
        st.info("✅ **Auto-save enabled!** Your progress is saved automatically.")
        st.write(f"📊 Data loaded: {'Yes' if st.session_state['df_original'] is not None else 'No'}")
        st.write(f"📊 Imbalance analyzed: {'Yes' if st.session_state['imbalance_info'] is not None else 'No'}")
        st.write(f"📊 Augmentation done: {'Yes' if st.session_state['df_combined'] is not None else 'No'}")
        
        if st.button("🗑️ Clear All Cache & Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            clear_cache()
            st.rerun()

    # Step 1: Load data from database
    st.subheader("📊 Step 1: Load Original Dataset")
    
    if st.button("🔄 Load Data from Database"):
        with st.spinner("Loading data from database..."):
            df_original = load_dataset()
            st.session_state['df_original'] = df_original
            save_to_cache('df_original', df_original)  # Save to disk
            st.success(f"✅ Loaded {len(df_original)} records from database")
            st.rerun()
    
    # Try to recover from cache if not in session
    if st.session_state['df_original'] is None:
        cached_df = load_from_cache('df_original')
        if cached_df is not None:
            st.session_state['df_original'] = cached_df
            st.info("♻️ Recovered data from cache!")
    
    if st.session_state['df_original'] is not None:
        df_original = st.session_state['df_original']
        st.dataframe(df_original.head(10))
        st.info(f"Total records: {len(df_original)}")

        # Step 2: Identify imbalanced classes
        st.subheader("⚖️ Step 2: Identify Imbalanced Classes")
        
        col1, col2 = st.columns(2)
        with col1:
            balance_column = st.selectbox("Column to check for imbalance", ['subject', 'status'])
        with col2:
            threshold_ratio = st.number_input("Imbalance threshold ratio", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        
        if st.button("🔍 Analyze Class Distribution"):
            imbalance_info = identify_imbalanced_classes(df_original, column=balance_column, threshold_ratio=threshold_ratio)
            st.session_state['imbalance_info'] = imbalance_info
            save_to_cache('imbalance_info', imbalance_info)  # Save to disk
            st.rerun()
        
        # Recover from cache
        if st.session_state['imbalance_info'] is None:
            cached_info = load_from_cache('imbalance_info')
            if cached_info is not None:
                st.session_state['imbalance_info'] = cached_info
                st.info("♻️ Recovered imbalance info from cache!")
        
        if st.session_state['imbalance_info'] is not None:
            imbalance_info = st.session_state['imbalance_info']
            
            # Display class distribution
            st.subheader(f"Class Distribution - {imbalance_info['column'].capitalize()}")
            
            dist_data = []
            for cls, info in imbalance_info['class_info'].items():
                dist_data.append({
                    'Class': cls,
                    'Count': info['count'],
                    'Ratio (Max/Current)': f"{info['ratio']:.2f}x",
                    'Status': '🔴 Minority (Need Augmentation)' if info['is_minority'] else '🟢 Majority'
                })
            
            dist_df = pd.DataFrame(dist_data)
            st.dataframe(dist_df, use_container_width=True)
            
            if imbalance_info['minority_classes']:
                st.warning(f"⚠️ Found {len(imbalance_info['minority_classes'])} minority class(es): {', '.join(map(str, imbalance_info['minority_classes']))}")
                
                # Step 3: Augmentation settings
                st.subheader("⚙️ Step 3: Augmentation Settings")
                col1, col2 = st.columns(2)
                with col1:
                    num_aug = st.number_input("Number of augmentations per text", min_value=1, max_value=10, value=2)
                with col2:
                    ks_threshold = st.number_input("K-S p-value threshold", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

                minority_classes = imbalance_info['minority_classes']
                balance_column = imbalance_info['column']
                df_minority = df_original[df_original[balance_column].isin(minority_classes)]
                
                st.info(f"🎯 Will augment {len(df_minority)} records from minority classes")

                # Step 3.5: Validate minority class data
                st.subheader("✅ Step 3.5: Validate Minority Class Data Quality (K-S Test)")
                st.markdown("**Method:** Validate internal consistency by comparing first half vs second half of minority data")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Why this method?**")
                    st.markdown("- Validates ALL minority records")
                    st.markdown("- Checks if data is internally consistent")
                    st.markdown("- No need for majority class comparison")
                with col2:
                    st.markdown("**Interpretation:**")
                    st.markdown("- p > 0.05: ✅ Data is consistent")
                    st.markdown("- p < 0.05: ⚠️ Data has inconsistencies")
                
                if st.button("🔍 Validate ALL Minority Class Data"):
                    with st.spinner(f"🔍 Validating all {len(df_minority)} minority records..."):
                        validation_result = validate_minority_data_quality(df_minority)
                    
                    st.session_state['minority_validation'] = validation_result
                    save_to_cache('minority_validation', validation_result)
                    st.rerun()
                
                # Recover validation
                if st.session_state['minority_validation'] is None:
                    cached_val = load_from_cache('minority_validation')
                    if cached_val is not None:
                        st.session_state['minority_validation'] = cached_val
                
                if st.session_state['minority_validation'] is not None:
                    validation_result = st.session_state['minority_validation']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("K-S Statistic", f"{validation_result['ks_stat']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{validation_result['p_value']:.4f}")
                    with col3:
                        status = "✅ Valid" if validation_result['is_valid'] else "⚠️ Warning"
                        st.metric("Status", status)
                    
                    if validation_result['is_valid']:
                        st.success(f"✅ {validation_result['interpretation']}")
                        st.info("👍 All minority data is consistent. Safe to proceed with augmentation!")
                    else:
                        st.warning(f"⚠️ {validation_result['interpretation']}")
                        st.info("💡 This might indicate:\n"
                               "- Mixed data sources in minority class\n"
                               "- Different writing styles within minority data\n"
                               "- Data quality variations\n\n"
                               "You can still proceed, but review the minority class data manually.")
                    
                    st.info(f"📝 Validated {validation_result['total_samples']} minority records\n"
                           f"- First half: {validation_result['first_half_samples']} samples\n"
                           f"- Second half: {validation_result['second_half_samples']} samples")
                    
                    # Button to save validation results to database
                    if st.button("💾 Save Validation Results to Database"):
                        try:
                            save_minority_validation_to_db(
                                validation_result, 
                                minority_classes, 
                                balance_column
                            )
                            st.success("✅ Pre-augmentation validation results saved to 'pre_augmentation_validation' table!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error saving to database: {str(e)}")

                # Step 4: Augment
                if st.session_state['minority_validation'] is not None or st.checkbox("⚠️ Skip validation"):
                    if st.button("🚀 Start Augmentation & Validation"):
                        augmented_rows = []
                        validation_results = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        total_rows = len(df_minority)
                        
                        for idx, (_, row) in enumerate(df_minority.iterrows()):
                            status_text.text(f"Processing {idx + 1}/{total_rows}: {row['title'][:50]}...")
                            
                            try:
                                aug_texts = augment_text_with_bert(row["text"], num_aug=num_aug)
                                
                                for aug_text in aug_texts:
                                    is_valid, ks_result = ks_validate(row["text"], aug_text, threshold=ks_threshold)
                                    
                                    validation_results.append({
                                        'is_valid': is_valid,
                                        'ks_stat': ks_result['ks_stat'],
                                        'p_value': ks_result['p_value']
                                    })
                                    
                                    if is_valid:
                                        augmented_rows.append({
                                            "title": row["title"],
                                            "text": aug_text,
                                            "subject": row["subject"],
                                            "status": row["status"],
                                            "is_augmented": True
                                        })
                            
                            except Exception as e:
                                st.warning(f"⚠️ Error: {str(e)}")
                            
                            progress_bar.progress((idx + 1) / total_rows)
                            
                            # Save progress periodically
                            if idx % 10 == 0:
                                save_to_cache('augmented_rows_progress', augmented_rows)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Combine data
                        df_original_flagged = df_original.copy()
                        df_original_flagged['is_augmented'] = False
                        df_augmented = pd.DataFrame(augmented_rows)
                        
                        if len(df_augmented) > 0:
                            df_combined = pd.concat([df_original_flagged, df_augmented], ignore_index=True)
                        else:
                            df_combined = df_original_flagged
                        
                        st.session_state['df_combined'] = df_combined
                        save_to_cache('df_combined', df_combined)
                        
                        # Save stats
                        if validation_results:
                            validation_stats = {
                                'total_original': len(df_original),
                                'total_augmented_generated': len(validation_results),
                                'total_valid_augmented': len(df_augmented),
                                'total_invalid_augmented': len(validation_results) - len(df_augmented),
                                'avg_ks_stat': np.mean([v['ks_stat'] for v in validation_results]),
                                'avg_p_value': np.mean([v['p_value'] for v in validation_results])
                            }
                            st.session_state['validation_stats'] = validation_stats
                            save_to_cache('validation_stats', validation_stats)
                        
                        st.success("✅ Augmentation completed!")
                        st.rerun()
                
                # Display results if available
                if st.session_state['df_combined'] is not None:
                    st.subheader("📦 Step 4: Combined Dataset")
                    df_combined = st.session_state['df_combined']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Original", len(df_original))
                    with col2:
                        st.metric("Minority", len(df_minority))
                    with col3:
                        augmented_count = len(df_combined[df_combined['is_augmented'] == True])
                        st.metric("Augmented", augmented_count)
                    with col4:
                        st.metric("Total", len(df_combined))
                    
                    st.dataframe(df_combined.head(20))
                    
                    # Save to database
                    st.subheader("💾 Step 5: Save to Database")
                    if st.button("💾 Save Combined Dataset to Database"):
                        with st.spinner("Saving..."):
                            save_combined_dataset(df_combined)
                            if st.session_state['validation_stats']:
                                save_validation_stats(st.session_state['validation_stats'])
                            st.success("✅ Saved to database!")
                            st.balloons()
            else:
                st.success("✅ Dataset is balanced!")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    augment_page()