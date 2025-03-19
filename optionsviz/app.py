"""
Module Name: app.py

Description:
    This module serves as the entry point for the 
    Options Data Visualization application.
    It initializes the Streamlit app and sets the page configuration.

Author:
    Ryan J Richards

Created:
    02-20-2025

License:
    MIT
"""
import streamlit as st

for k, v in st.session_state.items():
    st.session_state[k] = v

st.set_page_config(
    page_title="Options Data Visualization",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([st.Page("eod_chain.py", title='EOD Chain'),
                    st.Page("strategy.py", title='Strategy Selection')])
pg.run()
