import streamlit as st

st.set_page_config(
page_title="Weather CorrDiff",
page_icon="🌤️",
layout="wide",
initial_sidebar_state="expanded"
)

st.title("🌤️ Weather CorrDiff")
st.write("Welcome. Use the sidebar to navigate pages (Raw Data, Regression, Diffusion, Visualization & Stats).")

with st.sidebar:
    st.header("Quick Links")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/raw_data.py", label="Raw Data", icon="🔎")
    st.page_link("pages/regression.py", label="Regression", icon ="📈")
    st.page_link("pages/diffusion.py", label="Diffusion", icon = "🎛️")
    st.page_link("pages/visualization.py", label="Visualization & Stats", icon="📊")

    st.info("Next: create the pages/* files and reload.")