# Import Libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

# Streamlit Page Configuration
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Step 3: Load PDF Document
def load_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Step 7: Initialize Groq Chatbot
def ask_question(question, knowledge_base, model_llm, groq_api_key, chat_history):
    docs = knowledge_base.similarity_search(question, 5)
    llm = ChatGroq(groq_api_key=groq_api_key, model=model_llm)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Include chat history in the prompt
    chat_history_text = "\n".join([f"Q: {msg['content']}" if msg["role"] == "user" else f"A: {msg['content']}" for msg in chat_history if msg["role"] in ["user", "assistant"]])
    prompt = f"{chat_history_text}\n\nQ: {question}\nA:"
    
    answer = chain.run(input_documents=docs, question=question)  # Updated to use `question`
    return answer

# Main Streamlit App
def main():
    st.sidebar.title("PDF Chatbot")
    
    # Step 1: Ask User to Upload PDF
    pdf_file = st.sidebar.file_uploader("Upload your PDF document", type="pdf")

    # Model selection
    models = {
        'multi-qa-MiniLM-L6-cos-v1': 'multi-qa-MiniLM-L6-cos-v1',
        'intfloat/multilingual-e5-small': 'intfloat/multilingual-e5-small',
        'BAAI/bge-m3': 'BAAI/bge-m3',
    }
    model_name = st.sidebar.selectbox('Select Embedding Model', list(models.keys()))

    if pdf_file and model_name:
        if st.sidebar.button("Process"):
            with st.spinner("Processing PDF..."):
                document_text = load_pdf(pdf_file)
                st.session_state.document_text = document_text
                st.session_state.document_name = pdf_file.name

                # Step 2: Split Document into Chunks
                chunk_size = 256 * 5  # Adjust the chunk size as needed
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=150,
                    length_function=len
                )
                chunks = text_splitter.split_text(document_text)
                st.session_state.chunks = chunks

                # Step 3: Create Embeddings
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                st.session_state.knowledge_base = knowledge_base

                st.session_state.qa_pairs = []  # Initialize chat history

    if "document_name" in st.session_state:
        st.title(f"Chat with {st.session_state.document_name}")
    else:
        st.title("Chat with PDF")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # If knowledge base exists, use it to generate a response
        if "knowledge_base" in st.session_state:
            response = ask_question(prompt, st.session_state.knowledge_base, 'llama3-70b-8192', "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", st.session_state.messages)
        else:
            response = f"Echo: {prompt}"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Run the Streamlit App
if __name__ == "__main__":
    main()
