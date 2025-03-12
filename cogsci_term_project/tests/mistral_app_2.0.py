import os
import chainlit as cl
from mistralai import Mistral
import fitz  # PyMuPDF
from fastapi import FastAPI, Depends
from starlette.staticfiles import StaticFiles
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from typing import Any

app = FastAPI()

# Specify the directory for static files
static_directory = "static"

# Check if the directory exists, if not, create it
if not os.path.exists(static_directory):
    os.makedirs(static_directory)

# Mount the static files directory
app.mount("/Users/rbenner/Documents/GitHub/cogsci_term_project/static/sample.pdf", StaticFiles(directory=static_directory), name="static")

# Initialize the Mistral client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# ChromaDB connection setup
def get_chroma_client() -> chromadb.Client:
    # Use the default or supported API implementation
    return chromadb.Client()

@cl.on_chat_start
async def start_chat():
    # Send an initial greeting message to the user
    await cl.Message(content="🤖 Hello! Welcome to CogSci chat. How can I assist you today? 🧠🔬").send()

def extract_text_from_pdf(pdf_path: str) -> str:
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = "static/sample.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# Log the extracted text to verify
print("Extracted PDF Text:", pdf_text[:500])  # Print the first 500 characters for verification

# Placeholder function for document retrieval
async def retrieve_documents(query: str, chroma_client: chromadb.Client = Depends(get_chroma_client)) -> str:
    # Ensure the client is correctly initialized
    if chroma_client is None:
        raise ValueError("ChromaDB client is not correctly initialized.")

    collection_name = "cogsci"

    try:
        # Try to get the collection
        collection = chroma_client.get_collection(name=collection_name)
    except InvalidCollectionException:
        # If the collection does not exist, create it
        collection = chroma_client.create_collection(name=collection_name)

    # Add the extracted PDF text to the collection if not already present
    if not collection.count():
        collection.add(documents=[pdf_text], metadatas=[{"source": "sample.pdf"}], ids=["pdf_doc"])

    # Perform a query to retrieve documents
    documents = collection.query(query_texts=[query])  # Adjust the query as needed
    # Process the documents and return the relevant information
    retrieved_info = "\n".join([doc for doc in documents["documents"][0]])  # Adjust field name as needed
    return retrieved_info

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve relevant documents based on the user's query
    chroma_client = get_chroma_client()
    retrieved_info = await retrieve_documents(message.content, chroma_client)

    # Combine the retrieved information with the user's query
    augmented_query = f"User query: {message.content}\nRelevant information: {retrieved_info}"

    response = await client.chat.complete_async(
        model="mistral-small-latest",
        max_tokens=100,
        temperature=0.00,
        stream=False,
        messages=[
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": augmented_query  # Use the augmented query
            }
        ]
    )
    await cl.Message(content=response.choices[0].message.content).send()
