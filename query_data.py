import os
import argparse
import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from populate_database import db_update

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)



def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # st.write(formatted_response)
    return formatted_response


# if __name__ == "__main__":
#     main()



st.title("Fedex Chatbot Assistant")

uploaded_file = st.file_uploader("upload a pdf file", type = ['pdf'])

if uploaded_file:
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    db_update()

clear_db = st.button("Clear DB")

if clear_db:
    db_update(args_reset_flag=True)

prompt = st.text_area("Enter your prompt")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response ...."):
            st.write(query_rag(prompt))
