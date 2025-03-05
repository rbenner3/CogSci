import os
import chainlit as cl
from deepseek_app import Deepseek  # Import Deepseek client
import fitz  # PyMuPDF
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

app = FastAPI()

# Specify the directory for static files
static_directory = "static"

# Check if the directory exists, if not, create it
if not os.path.exists(static_directory):
    os.makedirs(static_directory)

# Mount the static files directory
app.mount("/Users/rbenner/Documents/GitHub/cogsci_term_project/static/sample.pdf", StaticFiles(directory=static_directory), name="static")

# Initialize the Deepseek client
client = Deepseek(api_key=os.getenv("DEEPSEEK_API_KEY"))

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

# Placeholder function for document retrieval
async def retrieve_documents(query: str) -> str:
    # Simple retrieval: return the entire PDF text
    # In practice, implement a more sophisticated retrieval mechanism
    return pdf_text

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve relevant documents based on the user's query
    retrieved_info = await retrieve_documents(message.content)

    # Combine the retrieved information with the user's query
    augmented_query = f"User query: {message.content}\nRelevant information: {retrieved_info}"

    response = await client.chat.complete_async(
        model="deepseek-small-latest",  # Update with the appropriate Deepseek model
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
