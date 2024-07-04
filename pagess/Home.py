import streamlit as st
from streamlit_chat import message as st_chat_message
from src.chatbot import chat
import openai
import time
from src.global_state import global_state



import os
import streamlit as st
from openai import OpenAI

def app():
    if 'user' not in st.session_state or st.session_state.user is None:
        st.write("Please log in to continue.")

    else:
        st.markdown("""<style>.block-container{max-width: 66rem !important;}</style>""", unsafe_allow_html=True)
        st.title("UG Buddy Demo")
        st.markdown('---')
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'streaming' not in st.session_state:
            st.session_state.streaming = False
        openai_key = st.secrets["OPENAI_API_KEY"]
        print(openai_key)
        if openai_key is None:
            with st.sidebar:
                st.subheader("Settings")
                openai_key = st.text_input("Enter your OpenAI key:", type="password")
        elif openai_key:
            chat()
        else:
            st.error("Please enter your OpenAI key in the sidebar to start.")
