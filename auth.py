import streamlit as st

def check_auth():
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Already logged in
    if st.session_state.authenticated:
        return True

    # Login UI
    st.title("VEEP Prospect Tool")
    st.markdown("#### Please enter the access password")

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password..."
    )

    if st.button("Login"):
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False