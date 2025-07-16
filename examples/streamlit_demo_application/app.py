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


from streamlit_folium import st_folium, folium_static
import folium

m = folium.Map(
    location=[47.6061, -122.3328],
    zoom_start=8,
    control_scale=True,
    # tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    # attr="Google Satellite",
    tiles="Carto DB darkmatter",
)

st_data = st_folium(m, use_container_width=True)
# st.map(latitude=47.6061, longitude=122.3328, zoom=10, use_container_width=True)
