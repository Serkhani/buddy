import streamlit as st
from streamlit_option_menu import option_menu
import pagess.Home as home
import pagess.Account as account

st.set_page_config(
        page_title="UG Buddy",
        page_icon=":pencil:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
st.logo('assets/ug_logo.png')

def main():
    selected = None
    with st.sidebar:
        selected = option_menu("UG Buddy", ["Home", 'Account'], 
            icons=['house', 'account'], menu_icon="cast", default_index=0)
        if 'user' in st.session_state and st.session_state.user is not None:
            if st.button("Logout"):
                st.session_state.user = None
    if selected == "Home":
        home.app()
    elif selected == "Account":
        account.app()

if __name__ == "__main__":
    main()