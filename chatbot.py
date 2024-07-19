# Install Required Libraries
# !pip install PyPDF2 langchain langchain-community langchain-groq groq FAISS-cpu sentence-transformers streamlit

# Import Libraries
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

# Streamlit Page Configuration
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Hide warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Step 3: Load PDF Document
def load_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Step 7: Initialize Groq Chatbot
def ask_question(question, knowledge_base, model_llm, groq_api_key):
    docs = knowledge_base.similarity_search(question, 5)
    llm = ChatGroq(groq_api_key=groq_api_key, model=model_llm)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    return answer

# Main Streamlit App
def main():
    query_params = st.experimental_get_query_params()
    if 'page' not in query_params:
        st.experimental_set_query_params(page=['upload'])
        query_params = st.experimental_get_query_params()

    if query_params['page'][0] == 'upload':
        st.title("PDF Chatbot - Upload and Model Selection")

        # Step 1: Ask User to Upload PDF
        pdf_file = st.file_uploader("Upload your PDF document", type="pdf")

        # Model selection
        models = {
            'multi-qa-MiniLM-L6-cos-v1': 'multi-qa-MiniLM-L6-cos-v1',
            'intfloat/multilingual-e5-small': 'intfloat/multilingual-e5-small',
            'BAAI/bge-m3': 'BAAI/bge-m3',
        }
        model_name = st.selectbox('Select Embedding Model', list(models.keys()))

        if pdf_file and model_name:
            if st.button("Process"):
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

                    # Redirect to chat page
                    st.experimental_set_query_params(page=['chat'])
                    st.experimental_rerun()

    elif query_params['page'][0] == 'chat':
        st.title(f"Ask any questions from the document: {st.session_state.document_name}")
        groq_api_key = "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"
        model_llm = 'llama3-70b-8192'  # Fixed model for simplicity


        # Display previous questions and answers
        if 'qa_pairs' not in st.session_state:
            st.session_state.qa_pairs = []

        for qa in st.session_state.qa_pairs:
            st.write(f"**Question:** {qa['question']}")
            st.write(f"**Answer:** {qa['answer']}")

        # Step 4: Ask Questions
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input("Ask a question here", key="question_input")
        with col2:
            if st.button("Ask"):
                if question:
                    answer = ask_question(question, st.session_state.knowledge_base, model_llm, groq_api_key)
                    st.session_state.qa_pairs.append({'question': question, 'answer': answer})
                    st.experimental_rerun()

# Run the Streamlit App
if __name__ == "__main__":
    main()
