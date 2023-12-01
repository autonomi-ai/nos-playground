from typing import List

from PIL import Image
from nos.client import Client

def test_sdxl_turbo():

    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "sd-xl-turbo"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    response: List[Image.Image] = model.text2image(prompts=["astronaut on the moon, hdr, 4k"], num_inference_steps=1)
    assert isinstance(response, list)
