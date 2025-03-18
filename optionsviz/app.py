import streamlit as st

for k, v in st.session_state.items():
    st.session_state[k] = v

st.set_page_config(
    page_title="Options Data Visualization",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",

)

pg = st.navigation([st.Page("eod_chain.py", title='EOD Chain'), st.Page("options_viz.py", title='Strategy Selection')])
pg.run()