# 📄 News Summarizer + Q&A using Groq API

import os
import streamlit as st
from dotenv import load_dotenv
from newspaper import Article
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# --- Step 0: Load environment variables ---
load_dotenv()

# --- Step 1: Initialize LLM using Groq API ---
llm = ChatOpenAI(
    model="llama3-8b-8192",  # or "gemma-7b-it"
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.5,
    max_tokens=512,
)

# --- Step 2: Scrape article from URL ---
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        return None, f"Error: {str(e)}"

# --- Step 3: Summarize full article text ---
def summarize_article(text, language="English"):
    prompt_template = """
You are a skilled multilingual news analyst. Summarize the following news article clearly and concisely in {language}.
Focus on:
- Key events
- People/organizations involved
- Date/time/location (if relevant)
- Consequences or outcomes
- Tone: Objective and informative

Article:
\"{text}\"

Summary (in {language}, under 200 words):
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text", "language"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"text": text.strip(), "language": language})

# --- Step 4: Create Vector DB with embeddings ---
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db

# --- Step 5: Create Custom Retrieval Q&A Chain ---
def get_qa_chain(db, language):
    qa_prompt = PromptTemplate(
        input_variables=["context", "question", "language"],
        template="""
You are a multilingual assistant. Based on the following article context, answer the user's question in {language}.

Context:
{context}

Question: {question}

Answer (in {language}):
"""
    )
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    combine_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context"
    )
    return RetrievalQA(
        retriever=db.as_retriever(),
        combine_documents_chain=combine_chain
    )

# --- Step 6: Streamlit UI ---
st.set_page_config(page_title="News Summarizer & Q&A with Groq")
st.title("📰 News Summarizer + Q&A (Powered by Groq)")

# --- Language selection ---
language_options = ["English", "Hindi", "Telugu", "Spanish", "French", "German"]
selected_language = st.selectbox("Choose summary language:", language_options, index=0)

# --- Initialize session states ---
if "url_input" not in st.session_state:
    st.session_state.url_input = ""
if "last_url" not in st.session_state:
    st.session_state.last_url = ""

# --- URL Input ---
url_input = st.text_input("Enter a news article URL:", value=st.session_state.url_input)

# --- If new URL is entered ---
if url_input and (url_input != st.session_state.last_url):
    with st.spinner("🔍 Scraping article..."):
        title, content = scrape_article(url_input)

    if content.startswith("Error"):
        st.error(content)
    else:
        st.session_state.last_url = url_input
        st.session_state.title = title
        st.session_state.content = content

        with st.spinner("📝 Generating Summary..."):
            st.session_state.summary = summarize_article(content, language=selected_language)

        with st.spinner("📚 Building knowledge base..."):
            st.session_state.db = create_vectorstore(content)
            st.session_state.qa_chain = get_qa_chain(st.session_state.db, selected_language)

        st.session_state.url_input = ""  # ✅ Clear input after processing

# --- Display Summary ---
if "summary" in st.session_state:
    st.subheader(f"📰 {st.session_state.title}")
    st.success("Summary ready!")
    st.markdown("### ✨ Summary")
    st.write(st.session_state.summary)

    # --- Q&A Section ---
    st.markdown("### ❓ Ask questions about the article")
    user_question = st.text_input("Your question:")

    if user_question:
        with st.spinner("💡 Generating answer..."):
            docs = st.session_state.qa_chain.retriever.get_relevant_documents(user_question)

        inputs = {
            "input_documents": docs,
            "question": user_question,
            "language": selected_language
        }

        response = st.session_state.qa_chain.combine_documents_chain.invoke(inputs)
        st.markdown("**Answer:**")
        if isinstance(response, dict) and "output_text" in response:
            st.success(response["output_text"])
        else:
            st.error("❌ Unable to extract answer.")
