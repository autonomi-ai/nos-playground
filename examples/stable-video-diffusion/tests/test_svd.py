from typing import List, Iterable

from PIL import Image

from nos.client import Client

TEST_IMAGE_LINK = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true"

def test_svd():
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "stable-video-diffusion"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    from typing import Iterable
    response: Iterable[Image.Image] = model.image2video(image=TEST_IMAGE_LINK, _stream=True)

    frames = []
    for resp in response:
        frames.append(resp)
        
        assert resp is not None
        assert isinstance(resp, Image.Image)
