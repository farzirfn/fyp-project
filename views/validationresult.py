import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ================================
# PAGE CONFIG
# ================================

# ================================
# DB CONNECTION
# ================================
def create_connection():
    return mysql.connector.connect(
        host="localhost", 
        user="root", 
        password="", 
        database="fyp"
    )

# ================================
# LOAD VALIDATION RESULTS
# ================================
def load_pre_augmentation_validation():
    """Load pre-augmentation validation results from database"""
    conn = create_connection()
    query = """
        SELECT 
            id,
            balance_column,
            minority_classes,
            total_samples,
            first_half_samples,
            second_half_samples,
            ks_statistic,
            p_value,
            is_valid,
            interpretation,
            validated_at
        FROM pre_augmentation_validation
        ORDER BY validated_at DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_post_augmentation_validation():
    """Load post-augmentation validation results from database"""
    conn = create_connection()
    query = """
        SELECT 
            id,
            total_original,
            total_augmented_generated,
            total_valid_augmented,
            total_invalid_augmented,
            avg_ks_stat,
            avg_p_value,
            created_at
        FROM validation_results
        ORDER BY created_at DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ================================
# MAIN APP
# ================================
def main():
    st.title("📊 Data Augmentation Validation Results")
    st.markdown("View and analyze validation results from data augmentation process")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📋 Pre-Augmentation Validation", 
        "📈 Post-Augmentation Stats",
        "📊 Comparison & Analytics"
    ])
    
    # ================================
    # TAB 1: Pre-Augmentation Validation
    # ================================
    with tab1:
        st.subheader("🔍 Pre-Augmentation Validation Results")
        st.markdown("**Purpose:** Validate minority class data quality BEFORE augmentation")
        
        try:
            df_pre = load_pre_augmentation_validation()
            
            if len(df_pre) == 0:
                st.info("📭 No validation results found. Run the augmentation process first.")
            else:
                # Summary metrics
                st.markdown("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Validations", len(df_pre))
                with col2:
                    valid_count = df_pre['is_valid'].sum()
                    st.metric("Valid Results", valid_count)
                with col3:
                    invalid_count = len(df_pre) - valid_count
                    st.metric("Warning Results", invalid_count)
                with col4:
                    avg_p_value = df_pre['p_value'].mean()
                    st.metric("Avg P-Value", f"{avg_p_value:.4f}")
                
                # Detailed table
                st.markdown("### 📋 Detailed Results")
                
                # Format the dataframe for display
                display_df = df_pre.copy()
                display_df['is_valid'] = display_df['is_valid'].apply(
                    lambda x: "✅ Valid" if x else "⚠️ Warning"
                )
                display_df['ks_statistic'] = display_df['ks_statistic'].apply(lambda x: f"{x:.4f}")
                display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}")
                display_df['validated_at'] = pd.to_datetime(display_df['validated_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization: P-value distribution
                st.markdown("### 📈 P-Value Distribution")
                fig = px.histogram(
                    df_pre, 
                    x='p_value',
                    nbins=20,
                    title="Distribution of P-Values",
                    labels={'p_value': 'P-Value', 'count': 'Frequency'},
                    color_discrete_sequence=['#667eea']
                )
                fig.add_vline(x=0.05, line_dash="dash", line_color="red", 
                             annotation_text="Threshold (0.05)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualization: KS Statistic
                st.markdown("### 📉 K-S Statistic Over Time")
                fig2 = px.line(
                    df_pre, 
                    x='validated_at', 
                    y='ks_statistic',
                    title="K-S Statistic Trend",
                    markers=True,
                    color_discrete_sequence=['#764ba2']
                )
                st.plotly_chart(fig2, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading pre-augmentation validation: {str(e)}")
    
    # ================================
    # TAB 2: Post-Augmentation Stats
    # ================================
    with tab2:
        st.subheader("📈 Post-Augmentation Statistics")
        st.markdown("**Purpose:** Statistics from the augmentation and validation process")
        
        try:
            df_post = load_post_augmentation_validation()
            
            if len(df_post) == 0:
                st.info("📭 No augmentation results found. Complete the augmentation process first.")
            else:
                # Latest run summary
                latest = df_post.iloc[0]
                
                st.markdown("### 📊 Latest Augmentation Run")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Original Records", latest['total_original'])
                with col2:
                    st.metric("Generated", latest['total_augmented_generated'])
                with col3:
                    st.metric("Valid Augmented", latest['total_valid_augmented'])
                with col4:
                    validation_rate = (latest['total_valid_augmented'] / latest['total_augmented_generated'] * 100)
                    st.metric("Validation Rate", f"{validation_rate:.1f}%")
                
                # Detailed table
                st.markdown("### 📋 All Augmentation Runs")
                display_post = df_post.copy()
                display_post['validation_rate'] = (
                    display_post['total_valid_augmented'] / 
                    display_post['total_augmented_generated'] * 100
                ).apply(lambda x: f"{x:.1f}%")
                display_post['avg_ks_stat'] = display_post['avg_ks_stat'].apply(lambda x: f"{x:.4f}")
                display_post['avg_p_value'] = display_post['avg_p_value'].apply(lambda x: f"{x:.4f}")
                display_post['created_at'] = pd.to_datetime(display_post['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(display_post, use_container_width=True)
                
                # Visualization: Validation rate over time
                st.markdown("### 📈 Validation Rate Trend")
                df_post['validation_rate'] = (
                    df_post['total_valid_augmented'] / 
                    df_post['total_augmented_generated'] * 100
                )
                fig = px.line(
                    df_post, 
                    x='created_at', 
                    y='validation_rate',
                    title="Validation Success Rate Over Time",
                    markers=True,
                    color_discrete_sequence=['#667eea']
                )
                fig.update_yaxes(title_text="Validation Rate (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Pie chart: Valid vs Invalid
                st.markdown("### 🥧 Valid vs Invalid Distribution (Latest Run)")
                pie_data = pd.DataFrame({
                    'Status': ['Valid', 'Invalid'],
                    'Count': [latest['total_valid_augmented'], latest['total_invalid_augmented']]
                })
                fig_pie = px.pie(
                    pie_data, 
                    values='Count', 
                    names='Status',
                    title="Augmentation Validation Results",
                    color_discrete_sequence=['#667eea', '#f5576c']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading post-augmentation stats: {str(e)}")
    
    # ================================
    # TAB 3: Comparison & Analytics
    # ================================
    with tab3:
        st.subheader("📊 Comparison & Analytics")
        
        try:
            df_pre = load_pre_augmentation_validation()
            df_post = load_post_augmentation_validation()
            
            if len(df_pre) == 0 or len(df_post) == 0:
                st.info("📭 Need both pre and post augmentation data for comparison.")
            else:
                st.markdown("### 🔄 Pre vs Post Augmentation Metrics")
                
                # Compare average metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Pre-Augmentation (Minority vs Majority)")
                    pre_metrics = pd.DataFrame({
                        'Metric': ['Avg K-S Statistic', 'Avg P-Value', 'Validation Success Rate'],
                        'Value': [
                            f"{df_pre['ks_statistic'].mean():.4f}",
                            f"{df_pre['p_value'].mean():.4f}",
                            f"{(df_pre['is_valid'].sum() / len(df_pre) * 100):.1f}%"
                        ]
                    })
                    st.dataframe(pre_metrics, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### Post-Augmentation (Original vs Augmented)")
                    latest_post = df_post.iloc[0]
                    post_metrics = pd.DataFrame({
                        'Metric': ['Avg K-S Statistic', 'Avg P-Value', 'Validation Success Rate'],
                        'Value': [
                            f"{latest_post['avg_ks_stat']:.4f}",
                            f"{latest_post['avg_p_value']:.4f}",
                            f"{(latest_post['total_valid_augmented'] / latest_post['total_augmented_generated'] * 100):.1f}%"
                        ]
                    })
                    st.dataframe(post_metrics, use_container_width=True, hide_index=True)
                
                # Comparison chart
                st.markdown("### 📊 P-Value Comparison")
                comparison_data = pd.DataFrame({
                    'Stage': ['Pre-Augmentation'] * len(df_pre) + ['Post-Augmentation'] * len(df_post),
                    'P-Value': list(df_pre['p_value']) + list(df_post['avg_p_value'])
                })
                fig = px.box(
                    comparison_data,
                    x='Stage',
                    y='P-Value',
                    title="P-Value Distribution: Pre vs Post Augmentation",
                    color='Stage',
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                             annotation_text="Threshold (0.05)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                st.markdown("### 💡 Insights")
                
                pre_valid_rate = df_pre['is_valid'].sum() / len(df_pre) * 100
                post_valid_rate = (latest_post['total_valid_augmented'] / 
                                  latest_post['total_augmented_generated'] * 100)
                
                if pre_valid_rate > 80:
                    st.success("✅ Pre-augmentation validation shows good minority data quality (>80% pass rate)")
                else:
                    st.warning("⚠️ Pre-augmentation validation shows potential quality issues in minority data")
                
                if post_valid_rate > 70:
                    st.success("✅ High augmentation validation rate indicates good quality augmented data")
                else:
                    st.warning("⚠️ Lower augmentation validation rate - consider adjusting parameters")
                
        except Exception as e:
            st.error(f"❌ Error in comparison: {str(e)}")

# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    main()