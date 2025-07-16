# Import Modules
import streamlit as st

# Set Page Configuration
st.set_page_config(
    page_title="FindMyWhale - Orca Localization",
    page_icon=":material/near_me:",
    layout="wide",
)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.header(":material/near_me: FindMyWhale", divider="gray")


st.write("Test")
