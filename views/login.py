import streamlit as st
import mysql.connector

def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="fyp"
    )

def login_page():
    st.set_page_config(page_title="Admin Login", page_icon="🔑", layout="centered")

    # Modern CSS styling
    st.markdown("""
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main container */
        .main {
            padding-top: 3rem;
        }
        
        /* Login container */
        .login-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
            margin: 2rem auto;
            max-width: 450px;
        }
        
        /* Title styling */
        .login-title {
            text-align: center;
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .login-subtitle {
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        
        /* Input styling */
        .stTextInput input {
            border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 0.75rem;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.95);
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus {
            border-color: white;
            box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.3);
            background: white;
        }
        
        /* Button styling */
        .stButton button {
            background: white;
            color: #667eea;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin-top: 1rem;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            background: #f8f9fa;
        }
        
        /* Back button styling */
        .back-button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 0.5rem 1.5rem;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-block;
            text-decoration: none;
            margin-bottom: 2rem;
        }
        
        .back-button:hover {
            background: rgba(255, 255, 255, 0.3);
            border-color: white;
            transform: translateX(-3px);
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 12px;
            margin-top: 1rem;
        }
        
        /* Icon styling */
        .input-icon {
            font-size: 1.2rem;
            margin-right: 0.5rem;
        }
        
        /* Security info box */
        .security-info {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 2rem;
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.85rem;
        }
        
        /* Decorative elements */
        .login-decorator {
            position: absolute;
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            z-index: -1;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Back to Home button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home", key="back_home"):
            st.session_state.page = "home"
            st.rerun()
    
    # Centered login form
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create centered container
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    
    with col2:
        # Login container with gradient background
        st.markdown("""
        <div class="login-container">
            <div class="login-title">🔑 Admin Login</div>
            <div class="login-subtitle">Secure access to admin dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Input fields
        st.markdown("<br>", unsafe_allow_html=True)
        
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            label_visibility="collapsed",
            key="username_input"
        )
        st.caption("👤 Username")
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            label_visibility="collapsed",
            key="password_input"
        )
        st.caption("🔒 Password")
        
        # Login button
        if st.button("🚀 Login", key="admin_login", use_container_width=True):
            if not username or not password:
                st.warning("⚠️ Please enter both username and password.")
            else:
                with st.spinner("🔐 Authenticating..."):
                    try:
                        conn = create_connection()
                        cursor = conn.cursor(dictionary=True)

                        query = "SELECT password FROM user WHERE username = %s"
                        cursor.execute(query, (username,))
                        result = cursor.fetchone()

                        if result and password == result["password"]:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.page = "admin_home"
                            st.success("✅ Login successful! Redirecting...")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password. Please try again.")

                        cursor.close()
                        conn.close()
                    
                    except mysql.connector.Error as e:
                        st.error(f"❌ Database error: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
        
        # Security info
        st.markdown("""
        <div class="security-info">
            🔒 Your login credentials are encrypted and secure.<br>
            For security reasons, please don't share your password.
        </div>
        """, unsafe_allow_html=True)
        
        # Additional info
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Need Help?"):
            st.markdown("""
            **Forgot your password?**  
            Contact the system administrator to reset your password.
            
            **Having trouble logging in?**  
            - Make sure your username is correct
            - Check that Caps Lock is off
            - Ensure you have admin privileges
            
            **Security Tips:**
            - Don't share your credentials
            - Use a strong, unique password
            - Log out after each session
            """)

# Run the login page
if __name__ == "__main__":
    login_page()