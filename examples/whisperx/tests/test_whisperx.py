from nos.client import Client
from nos.logging import logger
from nos.test.utils import NOS_TEST_AUDIO, get_benchmark_audio  # noqa: F401


def test_whisperx_transcribe_audio_file():
    from typing import List

    from rich.pretty import pretty_repr

    # Create a client
    client = Client("[::]:50051")
    assert client is not None
    assert client.WaitForServer()
    assert client.IsHealthy()

    model_id = "whisperx"
    models: List[str] = client.ListModels()
    assert model_id in models

    # Whisper
    # model_id = "m-bain/whisperx-large-v2"
    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    # Uplaod local audio path to server and execute inference
    # on the remote path. Note that the audio file is deleted
    # from the server after the inference is complete via the
    # context manager.
    logger.debug(f"Uploading [filename={NOS_TEST_AUDIO}, size={NOS_TEST_AUDIO.stat().st_size / (1024 * 1024):.2f} MB]")
    with client.UploadFile(NOS_TEST_AUDIO) as remote_path:
        response = model.transcribe(path=remote_path)
    assert isinstance(response, dict)
    logger.debug(pretty_repr(response))
    assert isinstance(response["segments"], list)
    assert isinstance(response["word_segments"], list)
    for item in response["segments"]:
        assert "start" in item
        assert "end" in item
        assert "text" in item
    for item in response["word_segments"]:
        assert "word" in item
        # not all words have a start/end/score keys
