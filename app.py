import streamlit as st
from pages.login import login_page
from pages.upload import upload_page
from pages.admin import stats_page  
from pages.user import user_home
from pages.model import model_page
from pages.augment import augment_page
from pages.validationresult import main

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Init session state
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "user_home"

# ================================
# SHARED CSS FUNCTION
# ================================
def apply_custom_css():
    """Apply custom CSS that persists across all pages"""
    st.markdown("""
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Hide Streamlit's default page navigation */
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebarNav"] {display: none;}
        
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] .css-1d391kg {
            color: white;
        }
        
        /* Sidebar text color */
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Sidebar button styling */
        [data-testid="stSidebar"] button {
            color: white !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        
        [data-testid="stSidebar"] button:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-color: rgba(255, 255, 255, 0.5) !important;
        }
        
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: rgba(255, 255, 255, 0.2) !important;
        }
        
        /* Title styling */
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .main-subtitle {
            color: #666;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 3rem;
        }
        
        /* Navigation cards */
        .nav-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            height: 100%;
            cursor: pointer;
        }
        
        .nav-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }
        
        .nav-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .nav-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .nav-description {
            color: #666;
            font-size: 0.95rem;
        }
        
        /* Feature box */
        .feature-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply CSS on every page load
apply_custom_css()

# -------------------------------
# Header (always visible)
# -------------------------------
col1, col2 = st.columns([9,2])
with col1:
    st.markdown("## 📰 Fake News Detection System via Augmented")
    st.markdown("---")

with col2:
    if not st.session_state.logged_in:
        if st.button("🔑 Admin Login", key="header_login"):
            st.session_state.page = "login"
            st.rerun()

# -------------------------------
# Sidebar (buttons instead of dropdown)
# -------------------------------
if st.session_state.logged_in:
    with st.sidebar:
        st.markdown("## 🛠️ Admin Panel")
        st.markdown("---")
        
        if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.page == 'stats' else "secondary"):
            st.session_state.page = "stats"
            st.rerun()
            
        if st.button("📤 Upload", use_container_width=True, type="primary" if st.session_state.page == 'upload' else "secondary"):
            st.session_state.page = "upload"
            st.rerun()
            
        if st.button("⚙️ Train Model", use_container_width=True, type="primary" if st.session_state.page == 'model' else "secondary"):
            st.session_state.page = "model"
            st.rerun()
            
        if st.button("🔄 Augment Data", use_container_width=True, type="primary" if st.session_state.page == 'augment' else "secondary"):
            st.session_state.page = "augment"
            st.rerun()
            
        if st.button("📊 Validation Results", use_container_width=True, type="primary" if st.session_state.page == 'validationresult' else "secondary"):
            st.session_state.page = "validationresult"
            st.rerun()
            
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.page = "user_home"
            st.rerun()

# -------------------------------
# Routing
# -------------------------------
if st.session_state.page == "user_home":
    user_home()   
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "upload" and st.session_state.logged_in:
    upload_page()
elif st.session_state.page == "stats" and st.session_state.logged_in:
    stats_page()
elif st.session_state.page == "model" and st.session_state.logged_in:
    model_page()
elif st.session_state.page == "admin_home" and st.session_state.logged_in:
    stats_page()
elif st.session_state.page == "augment" and st.session_state.logged_in:
    augment_page()
elif st.session_state.page == "validationresult" and st.session_state.logged_in:  
    main()