import os
import chainlit as cl
from mistralai import Mistral
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore



# Initialize the Mistral client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Load your data and create an index
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)

# Set up the service context
service_context = ServiceContext.from_defaults()

# Create a query engine
query_engine = RetrieverQueryEngine(index, service_context=service_context)

@cl.on_message
async def on_message(message: cl.Message):
    # Query the index to retrieve relevant documents
    retrieved_docs = query_engine.retrieve(message.content)

    # Extract the content from the retrieved documents
    context = "\n".join([doc.text for doc in retrieved_docs])

    # Use the retrieved context to augment the prompt
    response = await client.chat.complete_async(
        model="mistral-small-latest",
        max_tokens=100,
        temperature=0.5,
        stream=False,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful bot, you always reply in English"
            },
            {
                "role": "user",
                "content": f"Given the context: {context}, answer the following: {message.content}"
            }
        ]
    )
    await cl.Message(content=response.choices[0].message.content).send()
