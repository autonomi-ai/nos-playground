import pytest
import sys

from nos.client import Client
from rich import print


@pytest.fixture(scope="session", autouse=True)
def client():
    # Create a client
    client = Client("[::]:50051")
    assert client.WaitForServer()
    yield client


SYSTEM_PROMPT = "You are NOS chat, a Llama 2 large language model (LLM) agent hosted by Autonomi AI."
MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

@pytest.mark.parametrize("model_id", MODELS)
def test_streaming_chat(client, model_id):

    # Load the llama chat model
    model = client.Module(model_id)

    # Chat with the model
    query = "What is the meaning of life?"

    print()
    print("-" * 80)
    print(f">>> Chatting with the model (model={model_id}) ...")
    print(f"[bold yellow]Query: {query}[/bold yellow]")
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": query},
    ]
    for response in model.chat(messages=messages, max_new_tokens=1024, _stream=True):
        sys.stdout.write(response)
        sys.stdout.flush()
    print()
