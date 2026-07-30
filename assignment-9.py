import importlib
import os
import subprocess
import sys

# ==========================================
# 0. Automatic Dependency Installation
# ==========================================
# Explicit mapping of PyPI package names to actual Python import names
REQUIRED_PACKAGES = {
    "streamlit>=1.30.0": "streamlit",
    "langchain>=0.2.0": "langchain",
    "langchain-community>=0.2.0": "langchain_community",
    "langchain-huggingface>=0.0.1": "langchain_huggingface",
    "langchain-openai>=0.1.0": "langchain_openai",
    "faiss-cpu>=1.8.0": "faiss",  # Imported as 'faiss', NOT 'faiss_cpu'
    "pypdf>=4.0.0": "pypdf",
    "sentence-transformers>=2.5.0": "sentence_transformers",
    "python-dotenv>=1.0.0": "dotenv",  # Imported as 'dotenv', NOT 'python_dotenv'
}


def install_packages():
    """Checks and installs required packages dynamically at runtime."""
    installed_any = False
    for pkg_req, module_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"Installing missing dependency: {pkg_req}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg_req]
            )
            installed_any = True

    if installed_any:
        # Invalidate import caches so Python immediately recognizes the new modules
        importlib.invalidate_caches()


install_packages()

# ==========================================
# 1. Imports
# ==========================================
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (API keys)
load_dotenv()

# ==========================================
# 2. Page Configuration
# ==========================================
st.set_page_config(
    page_title="RAG Chatbot Capstone", page_icon="🤖", layout="wide"
)

st.title("🤖 Enterprise RAG Chatbot")
st.caption(
    "Upload your custom PDF documents and query them with grounded context."
)

# ==========================================
# 3. Sidebar Configuration & Setup
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Choose Model Provider
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
    )

    st.divider()
    st.subheader("📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Files", type=["pdf"], accept_multiple_files=True
    )

    process_btn = st.button("Process & Build Vector Index")


# ==========================================
# 4. Helper Functions for RAG Pipeline
# ==========================================
@st.cache_resource
def get_embedding_model():
    """Loads lightweight local HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_documents(files):
    """Extracts text, splits into chunks, and indexes into FAISS vector database."""
    documents = []

    for file in files:
        # Save uploaded file temporarily to process with PyPDFLoader
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs)
        os.remove(tmp_path)

    # Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Vector Database creation
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ==========================================
# 5. Vector Index Build Trigger
# ==========================================
if process_btn and uploaded_files:
    if not openai_api_key:
        st.sidebar.error("Please enter your OpenAI API key to continue.")
    else:
        with st.spinner("Extracting text & building FAISS index..."):
            try:
                st.session_state.vectorstore = process_documents(
                    uploaded_files
                )
                st.session_state.chain_ready = True
                st.sidebar.success("Vector store built successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing files: {str(e)}")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ==========================================
# 6. Chat Query Processing
# ==========================================
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    if not st.session_state.get("chain_ready", False):
        st.warning(
            "Please upload PDF documents and click 'Process & Build Vector Index' first."
        )
    else:
        # Append User Message to UI
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching document context & generating response..."
            ):
                # Define Prompt Template
                system_prompt = (
                    "You are a helpful assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise.\n\n"
                    "Context:\n{context}"
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # Setup Retriever & LLM
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )

                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.0,
                    openai_api_key=openai_api_key,
                )

                # Create LangChain Retrieval Chain
                question_answer_chain = create_stuff_documents_chain(
                    llm, prompt
                )
                rag_chain = create_retrieval_chain(
                    retriever, question_answer_chain
                )

                # Execute Chain
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                sources = response.get("context", [])

                st.write(answer)

                # Expandable Source Attribution
                with st.expander("📚 View Retrieved Document Contexts"):
                    for idx, doc in enumerate(sources):
                        st.markdown(
                            f"**Source {idx+1} (Page {doc.metadata.get('page', 'N/A') + 1}):**"
                        )
                        st.caption(doc.page_content[:300] + "...")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
