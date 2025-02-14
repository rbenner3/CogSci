import os
import chainlit as cl
from mistralai import Mistral

# Initialize the Mistral client
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

@cl.on_message
async def on_message(message: cl.Message):
    response = await client.chat.complete_async(
        model="mistral-small-latest",
        max_tokens=100,
        temperature=0.5,
        stream=False,
        # ... more setting
        messages=[
            {
                "role": "system",
                "content": "You are a helpful bot, you always reply in English"
            },
            {
                "role": "user",
                "content": message.content # Content of the user message
            }
        ]
    )
    await cl.Message(content=response.choices[0].message.content).send()
