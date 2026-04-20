import streamlit as st

def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # UI
    st.title("VEEP Prospect Tool")
    st.markdown("#### Please enter the access password")

    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == st.secrets.get("APP_PASSWORD", ""):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    # 🚨 IMPORTANT: do NOT just return False silently
    st.stop()