import os
import re
import random
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.ollama import Ollama


DATA_PATH = "data"
load_dotenv()


def get_vectorstore_from_pdf(pdf):
    # get the text in document form
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    document = document_loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    # vector_store = Chroma.from_documents(document_chunks, OllamaEmbeddings(model="nomic-embed-text"))
    vector_store = Chroma.from_documents(document_chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    return vector_store





### Rag Setup
def get_context_retriever_chain(vector_store):
    # llm = ChatOpenAI()
    # llm = Ollama(model="mistral")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, please answer if the question is relevant to the document.If you are unable to find the information in the document say you don't know.")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain






def get_conversational_rag_chain(retriever_chain): 
    # llm = ChatOpenAI()
    # llm = Ollama(model="mistral")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Please answer the following question using only information available in the document context.\
        Only answer if the question is directly relevant to the document's content. \
       If the question is unrelated to the document, reply politely with, \
       'I'm here to help with information specifically from the document. Let me know if you have a question related to its content, or please try to frame your question a bit differently. \
       Please have the text format consistent across the reponse:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)




def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    answer = response['answer']

    return answer




# app config
st.set_page_config(page_title="Chat with PDFs", page_icon="books")
st.title("Chat with PDFs")

# sidebar
with st.sidebar:
    st.header("Upload Files")
    uploaded_file = st.file_uploader("Please upload a pdf file", type = ['pdf'])

if uploaded_file is None:
    st.info("Welcome! 👋\
            I'm here to help you explore and understand your PDF documents. Simply upload a PDF file, and you can ask me questions about its contents—whether it's finding specific information, summarizing sections, or clarifying details.\
            Just upload a document to get started, and type your question below! 😊")
else:
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello !! How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_pdf(uploaded_file)    

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
            
        
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)