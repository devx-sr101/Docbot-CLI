import os
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
import argparse

sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = "data/"

def load_docs():
    docs = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())

            elif file.endswith(".txt"):
                loader = TextLoader(path)
                docs.extend(loader.load())

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    print(f"✅ Loaded {len(docs)} documents")
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks

DB_PATH = "db/"

def ingest():
    docs = load_docs()
    chunks = split_docs(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    print("✅ Ingestion complete!")


def ask(query):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the user's question based on the context below.\n\nContext:\n{context}\n\nQuestion:\n{query}"
    
    llm = ChatOllama(model="phi3")
    response = llm.invoke(prompt)

    print("\n💬 Answer:\n", response.content)

    print("\n📚 Sources:")
    for doc in docs:
        print("-", doc.metadata.get("source", "unknown"))

def main():
    parser = argparse.ArgumentParser(description="DocBot CLI")

    parser.add_argument("--ingest", action="store_true", help="Ingest documents")
    parser.add_argument("--ask", type=str, help="Ask a question")
    parser.add_argument("--watch", action="store_true", help="Watch folder for new files")

    args = parser.parse_args()

    if args.ingest:
        ingest()

    elif args.ask:
        ask(args.ask)

    elif args.watch:
        watch()

    else:
        print("❗ Use one of the following:")
        print("--ingest")
        print('--ask "your question"')
        print("--watch")

if __name__ == "__main__":
    main()