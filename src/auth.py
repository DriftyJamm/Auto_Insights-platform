import streamlit as st

# Dummy users (you can expand this)
users = {
    "admin": "admin123",
    "user": "user123"
}

def login():
    st.sidebar.subheader("🔐 Login")

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

def check_auth():
    return st.session_state.get("logged_in", False)