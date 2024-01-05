from typing import List

import cv2
from nos.client import Client


def test_dream_gaussian():
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "dream-gaussian"
    models: List[str] = client.ListModels()
    assert model_id in models

    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    test_img = cv2.imread("../csm_luigi_rgba.png", cv2.IMREAD_UNCHANGED)
    response = model(img=test_img)
    assert isinstance(response, dict)
    assert response["f"] is None
    assert response["v"] is None
    assert response["vt"] is None
    assert response["albedo"] is None
