# LangChain + Azure OpenAI + Streamlit Q&A App

Welcome to my tutorial that combines LangChain, Azure AI Foundry snd Azure AI services, and Streamlit to build a web application. This example will create a simple Q&A app that uses Azure OpenAI and LangChain for processing questions.

```python
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
```

To use this tutorial, follow these steps:

1. **Prerequisites Setup**

```bash
# Create a new directory
mkdir langchain-azure-streamlit
cd langchain-azure-streamlit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install langchain azure-openai streamlit python-dotenv
```

2. **Create .env file**
   Create a file named `.env` in your project directory with these variables:

```
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=your-azure-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

3. **Azure Setup**

- Create an Azure OpenAI resource in the Azure portal
- Get your API key, endpoint, and deployment name
- Update the `.env` file with your credentials

4. **Run the Application**

```bash
streamlit run app.py
```

**Tutorial Explanation:**

1. **Imports and Configuration**

- We import necessary libraries including Streamlit, LangChain components, and Azure OpenAI
- The `configure_azure_openai()` function sets up the Azure OpenAI environment variables

2. **LangChain Setup**

- `initialize_langchain()` creates:
  - An Azure OpenAI LLM instance
  - A custom prompt template
  - Conversation memory
  - A conversation chain combining all components

3. **Streamlit Interface**

- The main app includes:
  - A title and header
  - Chat message display
  - Input field for questions
  - Clear conversation button
- Uses Streamlit's session state to maintain conversation history

4. **Features**

- Maintains conversation history using LangChain's memory
- Displays messages in a chat-like interface
- Shows a loading spinner while processing
- Error handling for API calls
- Ability to clear conversation history

**To Enhance This App, You Could:**

1. Add file upload capability:

```python
uploaded_file = st.file_uploader("Upload a document")
if uploaded_file:
    # Add document processing logic
```

2. Add custom styling:

```python
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
```

3. Add more LangChain features:

```python
# Add document loader
from langchain.document_loaders import TextLoader
# Add vector store
from langchain.vectorstores import FAISS
# Add embeddings
from langchain.embeddings import OpenAIEmbeddings
```

This tutorial provides a foundation that you can build upon based on your specific needs. Make sure to:

- Handle your Azure credentials securely
- Monitor your Azure usage
- Add error handling as needed
- Test thoroughly before deployment

Let me know if you need help with any specific part of the implementation!
