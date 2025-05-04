import streamlit as st
from pages.capstone import show_capstone
from pages.churn import show_churn

# Set page configuration
st.set_page_config(
    page_title="Advanced ML Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # Changed to expanded for better navigation
)

# Initialize theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Apply custom styling with improved color schemes and consistency
def apply_custom_styling():
    if st.session_state.theme == 'dark':
        primary_color = "#00B875"  # Brighter green for better contrast in dark mode
        secondary_color = "#00956B"
        bg_color = "#121212"
        card_bg = "#1E1E1E"
        text_color = "#FFFFFF"
        secondary_text = "#B0B0B0"
        nav_bg = "#1A1A1A"
        border_color = "#333333"
        hover_color = "#00D68F"
        shadow = "0 4px 12px rgba(0, 0, 0, 0.3)"
    else:
        primary_color = "#00B875"
        secondary_color = "#00956B"
        bg_color = "#F8FAFB"  # Lighter background for better contrast
        card_bg = "#FFFFFF"
        text_color = "#333333"
        secondary_text = "#666666"
        nav_bg = "#FFFFFF"
        border_color = "#E6E6E6"
        hover_color = "#00D68F"
        shadow = "0 4px 12px rgba(0, 0, 0, 0.08)"

    # More professional and consistent styling
    st.markdown(f"""
    <style>
        /* Base styles */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
            font-family: 'Inter', sans-serif;
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5 {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }}
        
        /* Header */
        .dashboard-header {{
            font-size: 2.2rem;
            color: {primary_color};
            font-weight: 700;
            margin-bottom: 1.5rem;
            letter-spacing: -0.5px;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {primary_color};
        }}
        
        /* Cards */
        .dashboard-card {{
            border-radius: 12px;
            padding: 1.5rem;
            background-color: {card_bg};
            box-shadow: {shadow};
            margin-bottom: 1.5rem;
            border: 1px solid {border_color};
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .dashboard-card:hover {{
            transform: translateY(-2px);
            box-shadow: {shadow.replace('12px', '16px')};
        }}
        
        .card-title {{
            color: {primary_color};
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }}
        
        .card-text {{
            color: {secondary_text};
            font-size: 1rem;
            line-height: 1.5;
        }}
        
        /* Feature cards */
        .feature-card {{
            border-radius: 10px;
            padding: 1.2rem;
            background-color: {card_bg};
            box-shadow: {shadow};
            margin-bottom: 1rem;
            border: 1px solid {border_color};
            height: 100%;
        }}
        
        .feature-title {{
            color: {secondary_color};
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        /* Theme toggle button */
        .theme-toggle {{
            display: flex;
            justify-content: center;
            align-items: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: {primary_color};
            color: white;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
            transition: background-color 0.2s;
        }}
        
        .theme-toggle:hover {{
            background-color: {hover_color};
        }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-163ttbj {{ /* Streamlit sidebar classes */
            background-color: {nav_bg};
            border-right: 1px solid {border_color};
        }}
        
        /* Custom button styling */
        .custom-button {{
            background-color: {primary_color};
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            text-align: center;
            cursor: pointer;
            border: none;
            transition: background-color 0.2s;
            display: inline-block;
            margin-top: 1rem;
        }}
        
        .custom-button:hover {{
            background-color: {hover_color};
        }}
        
        /* Stats container */
        .stats-container {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        
        .stat-item {{
            background-color: {card_bg};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            box-shadow: {shadow};
            border: 1px solid {border_color};
        }}
        
        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {primary_color};
            margin-bottom: 0.5rem;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: {secondary_text};
        }}
    </style>
    """, unsafe_allow_html=True)

def render_home():
    st.markdown('<h1 class="dashboard-header">Advanced ML Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("""
    <div class="dashboard-card">
        <p class="card-text">
            This dashboard provides powerful machine learning tools to analyze your business data, 
            identify customer patterns, and make data-driven decisions. Select a tool from the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main features section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">📈 Customer Segmentation</div>
            <p class="card-text">
                Use unsupervised learning techniques to group customers based on behavior patterns.
                This tool helps businesses better understand their client base and target marketing efforts more efficiently.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🔄 Churn Prediction</div>
            <p class="card-text">
                Leverage predictive models to forecast which customers are likely to leave your service.
                This feature enables proactive customer retention strategies using historical data patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key benefits section
    st.markdown("<h2>Key Benefits</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Data-Driven Insights</div>
            <p class="card-text">Make business decisions based on comprehensive data analysis and visualization.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔍 Advanced Analytics</div>
            <p class="card-text">Leverage powerful machine learning algorithms for accurate predictive insights.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📱 Interactive Experience</div>
            <p class="card-text">Adjust parameters in real-time to explore different scenarios and outcomes.</p>
        </div>
        """, unsafe_allow_html=True)
    


def main():
    apply_custom_styling()
    
    # Using Streamlit's native sidebar for navigation
    with st.sidebar:
        st.markdown(f"<h3 style='text-align: center; color: {'#00B875' if st.session_state.theme == 'light' else '#00D68F'}'>Navigation</h3>", unsafe_allow_html=True)
        
        if st.button('🏠 Home', use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
            
        if st.button('📈 Customer Segmentation', use_container_width=True):
            st.session_state.page = 'capstone'
            st.rerun()
            
        if st.button('🔄 Churn Prediction', use_container_width=True):
            st.session_state.page = 'churn'
            st.rerun()
        
        # Adding spacing before theme toggle
        st.write("")
        st.write("")
        
        # Theme toggle in sidebar
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("Theme:")
        with col2:
            toggle_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
            if st.button(toggle_icon):
                st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
                st.rerun()
                
        # Version info
        st.write("")
        st.write("")
        st.write("")
        st.markdown(f"<p style='text-align: center; font-size: 0.8rem; color: {'#666666' if st.session_state.theme == 'light' else '#B0B0B0'}'>v1.0.0</p>", unsafe_allow_html=True)

    # Initialize page state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Render the correct page
    if st.session_state.page == 'home':
        render_home()
    elif st.session_state.page == 'capstone':
        show_capstone()
    elif st.session_state.page == 'churn':
        show_churn()

if __name__ == "__main__":
    main()