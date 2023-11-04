from nos.client import Client
from rich import print


SYSTEM_PROMPT = "You are NOS chat, a Llama 2 large language model (LLM) agent hosted by Autonomi AI."


def test_llama2_chat():

    # Create a client
    client = Client("[::]:50051")
    assert client.WaitForServer()

    # Load the llama chat model
    model = client.Module("meta-llama/Llama-2-7b-chat-hf")

    # Chat with the model
    query = "What is the meaning of life?"
    response = model.chat(message=query, system_prompt=SYSTEM_PROMPT)

    print(f"Query: {query}")
    for _i, r in enumerate(response):
        print(f"{r}")
