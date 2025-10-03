import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Better text splitting
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
import chromadb
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define paths
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_directory, "db", "chroma_db")

# This allows us to connect to and potentially manage the persistent collection
db_client = chromadb.PersistentClient(path=persistent_directory) 

# We check if the chroma vector store already exists
if not os.path.exists(persistent_directory) or not db_client.list_collections(): # check if directory exists AND if it has collections
    print("Persistent directory doesn't exist or is empty: Initializing the vector store")

    if not os.path.exists(file_path):
        print(f"The file {file_path} doesn't exist. Please check the path")
        exit()  # Exit if the document file is missing

    os.makedirs(persistent_directory, exist_ok=True) # Create the directory if it doesn't exist

    # Read the text content from the file 
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Display information about chunks
    print("\n------Document chunk information------")
    print(f"\nNumber of document chunks: {len(docs)}")
    print(f'\nSample chunk:\n{docs[0].page_content}\n')

    # Create a Gemini embedding function
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004" # Correct model name for Gemini embeddings
    )


    print("\n------Finished creating embedding function---------")

    # Create the vector store and persist it automatically
    print("\n--------Creating a vector store------")
    db = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory=persistent_directory)

    print("\n--------Vector store created and persisted------")
else:
    # If the vector store already exists, load it
    print("Vector store already exists. Loading it...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )
    db = Chroma(
        persist_directory=persistent_directory, 
        embedding_function=embeddings
    )

print("\nChromaDB vector store is ready!")

# Example: Query the vector store
query = "What is the One Ring's power?"
docs = db.similarity_search(query)
print("\n---Query Results---")
for doc in docs:
    print(doc.page_content)
