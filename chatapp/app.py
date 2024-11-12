# app.py
import streamlit as st
from inference import ask_question

# App Layout and Sidebar
st.set_page_config(page_title="VQA Chat", layout="wide")

# Sidebar for settings
st.sidebar.title("VQA Chat Settings")
st.sidebar.write("Ask questions about various objects in a conversational format.")

# Chat application
st.title("VQA Chat Assistant")
st.markdown("""
    <style>
        .chat-container {
            border: 1px solid #ddd;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            max-width: 700px;
            margin: auto;
        }
        .user-message, .bot-message {
            padding: 8px;
            border-radius: 10px;
            margin: 8px 0;
            width: fit-content;
        }
        .user-message {
            background-color: #dcf8c6;
            align-self: flex-end;
        }
        .bot-message {
            background-color: #fff;
            border: 1px solid #eee;
            align-self: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# Session State to store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-container"><div class="user-message">{message["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container"><div class="bot-message">{message["content"]}</div></div>', unsafe_allow_html=True)

# User input form
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Ask a question")
    object_name = st.text_input("Optional object name/context")
    submitted = st.form_submit_button("Send")

# Generate answer when form is submitted
if submitted and question:
    st.session_state["chat_history"].append({"role": "user", "content": question})

    # Call the inference model
    answer = ask_question(question, object_name)
    st.session_state["chat_history"].append({"role": "bot", "content": answer})

    # Display updated chat history
    st.rerun()
