import streamlit as st

class GlobalState:
    def __init__(self):
        self.user = None
        self.email = ''
        self.messages = []

if "global_state" not in st.session_state:
    st.session_state.global_state = GlobalState()

global_state = st.session_state.global_state
