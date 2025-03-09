# app.py

import streamlit as st
from langchain.llms import AzureOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Azure OpenAI settings
def configure_azure_openai():
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"  # Update as needed

# Initialize the LangChain components
def initialize_langchain():
    # Create the LLM with Azure OpenAI
    llm = AzureOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        model_name="gpt-35-turbo",  # Specify your model
        temperature=0.7
    )

    # Custom prompt template
    template = """You are a helpful AI assistant. Use the following conversation history and current input to provide a response.

Conversation history:
{history}

Current input: {input}

Response:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    # Initialize memory and conversation chain
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )

    return conversation

# Streamlit app
def main():
    # Configure Azure settings
    configure_azure_openai()

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = initialize_langchain()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Page setup
    st.title("Azure AI + LangChain Q&A App")
    st.subheader("Ask me anything!")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.predict(input=prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation = initialize_langchain()
        st.rerun()

if __name__ == "__main__":
    main()