import os
import streamlit as st
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Streamlit Page Config
st.set_page_config(page_title="Zyro HR Help Desk RAG", page_icon="🤖", layout="centered")

st.title("🤖 Zyro Dynamics HR Help Desk")
st.markdown("Ask any question regarding internal company HR policies.")

# Sidebar for API Key configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot uses RAG with LangChain, FAISS, HuggingFace embeddings, and Groq LLM to answer HR policy queries strictly based on company documents.")

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar to start chatting.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# Cache resource to load and embed PDFs only once
@st.cache_resource
def initialize_rag():
    pdf_paths = glob.glob("*.pdf") + glob.glob("policies/*.pdf")
    if not pdf_paths:
        pdf_paths = glob.glob("**/*.pdf", recursive=True)
    
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
    )
    
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
    
    template = """You are an expert HR Help Desk assistant for Zyro Dynamics Pvt. Ltd. 
Your task is to answer employee HR queries accurately and concisely using ONLY the provided context from internal company policy documents.

CRITICAL INSTRUCTIONS:
1. Ground your answer entirely in the provided context. Do not assume, extrapolate, or use outside knowledge.
2. If the user's question cannot be answered directly using the provided policy context, or if it is out-of-scope (e.g., general knowledge, personal questions, non-HR topics), you MUST refuse by responding with EXACTLY this sentence and nothing else:
"I'm sorry, I can only answer questions related to Zyro Dynamics internal HR policies."

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

with st.spinner("Loading HR policy documents and building vector store... Please wait."):
    chain = initialize_rag()

st.success("HR Policy Knowledge Base Loaded Successfully!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})