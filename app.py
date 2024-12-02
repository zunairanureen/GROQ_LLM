import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = "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"

def main():
    st.title("Groq Chat LLM")
    st.sidebar.title("Select a LLM model")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select a model",
        ["llama3-8b-8192", "mixtral-8x7b-32768", "whisper-large-v3"]
    )

    # Conversation memory length slider
    conservation_memory_length = st.sidebar.slider("Conversation memory length:", 2, 10, value=5)
    
    # Initialize or update memory and 'k' in session state
    if "memory" not in st.session_state or st.session_state.memory_k != conservation_memory_length:
        st.session_state.memory_k = conservation_memory_length
        st.session_state.memory = ConversationBufferMemory(k=conservation_memory_length)

    # User input
    user_question = st.text_area("User Question...", placeholder="Type your question here.")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sync chat history with memory context
    memory = st.session_state.memory
    for message in st.session_state.chat_history:
        memory.save_context({"input": message['human']}, {"output": message['AI']})

    # Initialize Groq Chat model
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    # ConversationChain with memory
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    # Handle user input
    if user_question:
        response = conversation.run(user_question)
        
        # Save interaction to chat history
        message = {"human": user_question, "AI": response}
        st.session_state.chat_history.append(message)

        # Display response
        st.write("Chat LLM:", response)

    # Show chat history
    st.sidebar.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.sidebar.write(f"**You:** {chat['human']}")
        st.sidebar.write(f"**AI:** {chat['AI']}")

if __name__ == "__main__":
    main()
