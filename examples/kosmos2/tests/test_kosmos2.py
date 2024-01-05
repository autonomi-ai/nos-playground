import requests
from nos.client import Client
from PIL import Image
from rich import print


TEST_PROMPTS = ["<grounding>An image of"]


def test_kosmos_2():
    # Create a client
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "microsoft/kosmos2"

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    prompt = "<grounding>An image of"

    url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # The original Kosmos-2 demo saves the image first then reload it. For some images, this will give slightly different image input and change the generation outputs.
    image.save("new_image.jpg")
    image = Image.open("new_image.jpg")

    response = model.image_to_text(prompt=prompt, image=image)
    assert isinstance(response, dict)
    assert response["processed_text"] is not None
    assert response["entities"] is not None

    processed_text = response["processed_text"]
    entities = response["entities"]
    print(f"Processed Text: {processed_text}")
    print(f"Entities: {entities}")
