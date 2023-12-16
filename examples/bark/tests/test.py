from typing import List

from nos.client import Client


def test_bark():
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "bark"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    response = model(prompt="Hi, this is an example from nos.")
    assert isinstance(response, dict)
    assert response["audio"] is not None
    assert response["sampling_rate"] is not None
