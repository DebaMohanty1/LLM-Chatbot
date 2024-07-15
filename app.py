import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

# Streamlit app title
st.title("PDF Summarizer")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Button to summarize
if uploaded_file is not None:
    if st.button("Summarize"):
        # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        # Load and split the PDF document
        loader = PyPDFLoader("temp.pdf")
        texts = loader.load_and_split()
        
        api_key = st.secrets['groq']['api-key']
        # Initialize the language model
        llm = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            api_key=api_key
        )
        
        # Load the summarize chain
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        
        # Summarize the document
        result = chain.invoke(texts)
        
        # Display the summary
        st.subheader("Summary:")
        st.write(result["output_text"])
