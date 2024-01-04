from nos.client import Client
import numpy as np
from torch import Tensor

TEST_SENTENCES = ["I'm happy", "I'm full of happiness"]
TEST_MODEL_ID =  "TaylorAI/gte-tiny"

def test_gte_embedding_embed():
    # Create a client
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = TEST_MODEL_ID
    
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    response = model.embed(text=TEST_SENTENCES[0])
    assert isinstance(response, dict)
    assert response["output"] is not None
    output = response["output"]
    assert isinstance(output, np.ndarray)


def test_gte_embedding_sentence_similarity():
    # Create a client
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = TEST_MODEL_ID
    
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    response = model.sentence_similarity(sentences = TEST_SENTENCES)
    assert isinstance(response, dict)
    assert response["output"] is not None
    output = response["output"]
    assert isinstance(output, Tensor)

