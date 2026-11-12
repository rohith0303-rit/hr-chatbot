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
st.set_page_config(
    page_title="Zyro Dynamics | Enterprise HR Help Desk", 
    layout="centered"
)

# Custom CSS Styling for Professional Polish
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        background-color: #F9FAFB;
        color: #1F2937;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #EFF6FF;
        border-color: #3B82F6;
        color: #1D4ED8;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<p class="main-header">Zyro Dynamics HR Help Desk</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Internal Policy Assistant powered by advanced RAG & LLMs</p>', unsafe_allow_html=True)

# Sidebar for Configuration & Quick FAQs
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    st.markdown("---")
    st.subheader("Model Settings")
    model_choice = st.selectbox(
        "Select LLM Model", 
        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"]
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=0.5, value=0.0, step=0.1)

    st.markdown("---")
    st.subheader("Quick HR FAQs")
    faq_selected = None
    if st.button("Maternity Leave Policy"):
        faq_selected = "What is our policy on maternity leave?"
    if st.button("Medical Insurance Coverage"):
        faq_selected = "What is the policy regarding health insurance coverage?"
    if st.button("Performance Review & PIP"):
        faq_selected = "When is the Annual Performance Review conducted and what triggers a PIP?"
    if st.button("Sick Leave & Medical Certificates"):
        faq_selected = "When is a medical certificate required for sick leave?"

    st.markdown("---")
    st.markdown("### About")
    st.markdown("Designed for Zyro Dynamics employees to query verified HR manuals with strict source grounding and automated refusals.")

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar to initialize the HR help desk.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# Cache resource to load and embed PDFs efficiently
@st.cache_resource
def initialize_rag(model_name, temp):
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
    
    llm = ChatGroq(model_name=model_name, temperature=temp)
    
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
    
    return retriever, prompt, llm

with st.spinner("Loading company policy corpus and building vector index..."):
    retriever, prompt_template, llm = initialize_rag(model_choice, temperature)

st.success("HR Knowledge Base Loaded Successfully!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Source Policy Citations"):
                for idx, src in enumerate(message["sources"], 1):
                    source_name = os.path.basename(src.metadata.get('source', 'Unknown Policy'))
                    page_num = src.metadata.get('page', 0) + 1
                    st.markdown(f"**Source {idx}:** `{source_name}` (Page {page_num})")
                    st.caption(src.page_content[:300] + "...")

# Determine user input (either from chat input box or sidebar FAQ buttons)
user_query = st.chat_input("Ask an HR policy question (e.g., probation period, leave rules)...")
if faq_selected:
    user_query = faq_selected

if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response with retrieved context & citations
    with st.chat_message("assistant"):
        with st.spinner("Analyzing HR guidelines..."):
            # Retrieve relevant document chunks
            retrieved_docs = retriever.invoke(user_query)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Format chain execution
            chain = (
                {"context": lambda x: context_text, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            response = chain.invoke(user_query)
            st.markdown(response)
            
            # Display source expander
            if retrieved_docs:
                with st.expander("View Source Policy Citations"):
                    for idx, src in enumerate(retrieved_docs, 1):
                        source_name = os.path.basename(src.metadata.get('source', 'Unknown Policy'))
                        page_num = src.metadata.get('page', 0) + 1
                        st.markdown(f"**Source {idx}:** `{source_name}` (Page {page_num})")
                        st.caption(src.page_content[:300] + "...")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response, 
        "sources": retrieved_docs
    })
