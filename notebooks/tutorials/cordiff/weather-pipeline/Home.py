import streamlit as st

st.set_page_config(
page_title="Weather CorrDiff",
page_icon="🌤️",
layout="wide",
initial_sidebar_state="expanded"
)

st.title("🌤️ Weather CorrDiff")
st.write("Welcome. Use the sidebar to navigate pages (Raw Data, Regression, Diffusion, Visualization & Stats).")

# with st.sidebar:
#     st.header("Quick Links")
#     st.page_link("Home.py", label="Home", icon="🏠")
#     st.page_link("pages/1_🔎 _Raw_Data.py", label="Raw Data", icon="🔎")
#     st.page_link("pages/2_📈_Regression.py", label="Regression", icon ="📈")
#     st.page_link("pages/3_🎛️_Diffusion.py", label="Diffusion", icon = "🎛️")
#     st.page_link("pages/4_📊_Statistics.py", label="Statistics", icon="📊")

#     st.info("Next: create the pages/* files and reload.")