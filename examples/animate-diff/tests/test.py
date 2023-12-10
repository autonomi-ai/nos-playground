from typing import List, Iterable
from PIL import Image
from nos.client import Client

def test_svd():
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "animate-diff"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    response: Iterable[Image.Image] = model(prompts=["astronaut on the moon, hdr, 4k"], _stream=True)
    for resp in response:
        assert resp is not None
        assert isinstance(resp, Image.Image)
