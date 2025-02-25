import streamlit as st

for k, v in st.session_state.items():
    st.session_state[k] = v

st.set_page_config(
    page_title="Options Data Visualization",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",

)

# pg = st.navigation([st.Page("single_chain.py", title='Single Chain'),
#                     st.Page("dynamic_chain.py", title='Dynamic Chain')])
pg = st.navigation([st.Page("dynamic_chain.py", title='Dynamic Chain')])
pg.run()