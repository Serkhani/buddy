import streamlit as st
from fb_streamlit_auth import fb_streamlit_auth
import os
from src.global_state import global_state

def app():
    # if global_state.user is None:
    if 'user' not in st.session_state or st.session_state.user is None:
        user = fb_streamlit_auth(
            st.secrets['API_KEY'],
            st.secrets['AUTH_DOMAIN'],
            st.secrets['DATABASE_URL'],
            st.secrets['PROJECT_ID'],
            st.secrets['STORAGE_BUCKET'],
            st.secrets['MESSAGING_SENDER_ID'],
            st.secrets['APP_ID'],
            st.secrets['MEASUREMENT_ID'],
            )
        if user is None:
            st.write("Please log in to continue.")
        else:
            # global_state.user = user
            st.session_state.user = user
            show_user_profile(user)
    else:
        # show_user_profile(global_state.user)
        show_user_profile(st.session_state.user)

def show_user_profile(user):
    st.markdown("<h1 style='text-align: center;'>Profile</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Welcome! {user['displayName']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Email: {user['email']}</h3>", unsafe_allow_html=True)
    if 'photoURL' in user:
        st.image(user['photoURL'], width=200, caption='Profile Picture')
