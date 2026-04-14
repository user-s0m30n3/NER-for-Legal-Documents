import streamlit as st
st.title("Streamlit Test")
st.write("If you see this, Streamlit rendering is working.")
st.sidebar.title("Sidebar Test")
uploaded = st.sidebar.file_uploader("Upload Test")
if uploaded:
    st.write("File uploaded!")
