import os
import streamlit as st
import pickle
import faiss
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("Generative AI Research Navigator 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_index.faiss"
docstore_path = "vector_index_docstore.pkl"
index_to_docstore_id_path = "vector_index_index_to_docstore_id.pkl"

main_placeholder = st.empty()
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    print("Loaded Data: ", data)
    
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print("Split Documents: ", docs)
    
    if not (os.path.exists(file_path) and os.path.exists(docstore_path) and os.path.exists(index_to_docstore_id_path)):
        # Create the embeddings of the chunks using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        print("Vector Index Created: ", vectorindex_openai)

        # Save the FAISS index to a file
        index = vectorindex_openai.index
        faiss.write_index(index, file_path)

        # Save the docstore and index_to_docstore_id separately
        with open(docstore_path, "wb") as f:
            pickle.dump(vectorindex_openai.docstore, f)

        with open(index_to_docstore_id_path, "wb") as f:
            pickle.dump(vectorindex_openai.index_to_docstore_id, f)

        embedding_function = vectorindex_openai.embedding_function
        main_placeholder.text("Embedding Vector Building Completed...✅✅✅")
    else:
        main_placeholder.text("FAISS index already exists. Skipping creation.")
        time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path) and os.path.exists(docstore_path) and os.path.exists(index_to_docstore_id_path):
        # Load the FAISS index from a file
        index = faiss.read_index(file_path)
        print("Loaded FAISS Index: ", index)

        # Load the docstore and index_to_docstore_id
        with open(docstore_path, "rb") as f:
            docstore = pickle.load(f)
        print("Loaded Docstore: ", docstore)

        with open(index_to_docstore_id_path, "rb") as f:
            index_to_docstore_id = pickle.load(f)
        print("Loaded Index to Docstore ID: ", index_to_docstore_id)

        # Recreate the embedding function if necessary
        embedding_function = OpenAIEmbeddings()

        # Recreate the FAISS object with the loaded index and additional data
        vectorIndex = FAISS(
            index=index,
            embedding_function=embedding_function,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
    else:
        st.error("FAISS index files do not exist. Please create the index first.")
