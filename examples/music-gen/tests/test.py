from typing import List

from nos.client import Client


def test_music_gen():
    client = Client()
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "music-gen"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    response = model(prompt="lo-fi music, chill beats to relax")
    assert isinstance(response, dict)
    assert response["audio"] is not None
    assert response["sampling_rate"] is not None
