from typing import List

from nos.client import Client
from PIL import Image


def test_sdxl_turbo():

    client = Client()
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

    response: List[Image.Image] = model(prompts=["astronaut on the moon, hdr, 4k"], num_inference_steps=1)
    assert isinstance(response, list)
